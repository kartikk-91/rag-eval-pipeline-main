# Infrastructure for the deployed demo.
#
# Shape: one Lambda function running the container image, reachable through an
# API Gateway HTTP API that invokes it via an IAM integration role. Not a Function
# URL: Lambda resource-based policies are not enforced on this account, so one
# cannot be made public (docs/decisions/009). No load balancer, no vector database
# — the index is baked into the image (docs/decisions/006), so there is nothing
# running, and nothing billed, between requests.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  name                = "rag-eval-pipeline"
  langfuse_ssm_prefix = "/${local.name}/langfuse"
}

# ----------------------------------------------------------------- ECR ------

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# The image is 2.4 GB. Without this, every deploy leaves the previous one behind
# at roughly €0.24/month each, forever.
resource "aws_ecr_lifecycle_policy" "keep_recent" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 3 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

# Lambda pulls the image as a service, not as the caller, so the execution role
# cannot grant this — it needs a resource policy on the repository itself.
# Without it CreateFunction fails with "Lambda does not have permission to access
# the ECR image", which names ECR without saying that a repository policy is what
# is missing. Scoped by sourceArn so only this function may pull.
resource "aws_ecr_repository_policy" "lambda_pull" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "LambdaECRImageRetrievalPolicy"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ]
      Condition = {
        StringLike = {
          "aws:sourceArn" = "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:${local.name}"
        }
      }
    }]
  })
}

# ----------------------------------------------------------------- IAM ------

resource "aws_iam_role" "lambda" {
  name = "${local.name}-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least privilege: invoke one model, in one region. Not bedrock:* on "*".
resource "aws_iam_role_policy" "bedrock" {
  name = "${local.name}-bedrock-invoke"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["bedrock:InvokeModel"]
      Resource = [
        # The "eu." prefix makes this a CROSS-REGION inference profile: Bedrock may
        # serve the request from any EU region, and authorises against the foundation
        # model in whichever one it picks. Pinning this to var.region passes locally
        # under AmazonBedrockFullAccess and then fails on the scoped role as soon as
        # the profile routes elsewhere. The profile itself stays pinned, so this does
        # not widen which model can be called — only where it may be served from.
        "arn:aws:bedrock:*::foundation-model/${var.bedrock_foundation_model}",
        "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.bedrock_model_id}",
      ]
    }]
  })
}

# Read the two Langfuse keys, and nothing else in Parameter Store. They are created
# by hand (infra/DEPLOY.md), not by Terraform: a secret Terraform writes is a secret
# in its state file. No kms:Decrypt here — SecureStrings under the AWS-managed
# aws/ssm key are decrypted on the caller's behalf by Parameter Store itself.
resource "aws_iam_role_policy" "langfuse_keys" {
  name = "${local.name}-langfuse-keys"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameters"]
      Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${local.langfuse_ssm_prefix}/*"
    }]
  })
}

# -------------------------------------------------------------- Lambda ------

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 14 # logs are not the product; do not accumulate them forever
}

resource "aws_lambda_function" "app" {
  function_name = local.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  # The embedding model loads on cold start and the index is copied to /tmp.
  # 2 GB buys proportionally more CPU, which is what actually shortens that.
  # The image's own CMD runs uvicorn, which is right for `docker run` and wrong for
  # Lambda: the runtime polls a client for events rather than speaking HTTP to the
  # container. Overriding the entry point here keeps one image serving both, instead
  # of maintaining a second Dockerfile that drifts from the first.
  image_config {
    entry_point = ["/usr/local/bin/python", "-m", "awslambdaric"]
    command     = ["api.lambda_handler.handler"]
  }

  # Peak usage is 718 MB, so this is not about RAM. Lambda scales CPU with memory and
  # the cold start is CPU-bound — importing torch and loading the embedding model —
  # so memory buys cores. Measured on this function, /health cold:
  #   2048 MB -> 62.1s     3008 MB -> 18.7s     (warm, either way: ~0.14s)
  # 3008 is the ceiling this account allows; the usual Lambda maximum is 10240.
  memory_size = 3008

  # A cold start is 18.7s measured, and a cold request also waits on Bedrock. 90s
  # leaves room for both without the 60s that actually timed out at 2048 MB.
  timeout = 90

  # The ceiling on a surprise bill. A public endpoint with unbounded concurrency is
  # an invitation. On an account whose total limit is 10 this must be -1 (see
  # variables.tf) and the account limit is the ceiling; the per-IP rate limit in the
  # application and the budget alarm are the other two layers.
  reserved_concurrent_executions = var.max_concurrency

  environment {
    variables = {
      QDRANT_BAKED_INDEX = "/opt/index"
      HF_HOME            = "/opt/hf"
      BEDROCK_MODEL_ID   = var.bedrock_model_id
      RATE_LIMIT_PER_MIN = tostring(var.rate_limit_per_min)
      # A path, not a key: the function fetches the keys from here at cold start.
      LANGFUSE_SSM_PREFIX = local.langfuse_ssm_prefix
    }
  }

  # The repository policy must exist before CreateFunction, which verifies the
  # image is pullable at create time rather than on first invoke.
  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_ecr_repository_policy.lambda_pull,
  ]
}

# The public face is API Gateway, and it invokes Lambda with an integration ROLE
# rather than a resource policy. That is not a style choice: on this account Lambda
# resource-based policies are not enforced at all. Measured on this function URL —
# a SigV4-signed request from a principal granted only via lambda:AddPermission
# returns 403, while the same request from a principal holding an identity-based
# grant returns 200; anonymous access with a Principal "*" grant returns 403, and so
# did CloudFront OAC, which depends on a resource-policy grant to its service
# principal. The account is in no Organization, so no SCP or RCP explains it.
#
# credentials_arn makes API Gateway assume this role and call Lambda with its
# credentials, so authorisation runs through an identity policy, which does work.
# There is no aws_lambda_permission anywhere below, and that absence is the point.
#
# No function URL either: API Gateway integrates with the function ARN directly.

resource "aws_iam_role" "apigw_invoke" {
  name = "${local.name}-apigw-invoke"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "apigateway.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "apigw_invoke" {
  name = "${local.name}-invoke"
  role = aws_iam_role.apigw_invoke.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["lambda:InvokeFunction"]
      Resource = [aws_lambda_function.app.arn]
    }]
  })
}

resource "aws_apigatewayv2_api" "app" {
  name          = local.name
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
    allow_headers = ["content-type"]
  }
}

resource "aws_apigatewayv2_integration" "app" {
  api_id                 = aws_apigatewayv2_api.app.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.app.arn
  payload_format_version = "2.0"
  credentials_arn        = aws_iam_role.apigw_invoke.arn

  # A cold start is 18.7s and a cold /ask also waits on Bedrock. The default is 30s,
  # and 30000 is the documented ceiling for HTTP APIs, so this is the most the
  # gateway will wait — which is why the Lambda timeout of 90s cannot be used in full.
  timeout_milliseconds = 30000
}

# $default catches every path, so FastAPI keeps doing the routing rather than having
# routes declared twice, in Python and in Terraform, free to disagree.
resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.app.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.app.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.app.id
  name        = "$default"
  auto_deploy = true
}
# ---------------------------------------------------------------- cost ------

# Lambda is free at idle, but "free" assumes nobody hammers a public endpoint.
# This is the tripwire, not the defence.
resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_eur
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
}

# ---------------------------------------------------------------- warm ------

# A cold /ask takes ~20s, most of it importing torch and loading the embedding model
# (it was ~47s before docs/decisions/017). HTTP API integrations cap at 30s and AWS
# does not allow that to be raised, so a cold start leaves little margin, and a
# slower one would reach the visitor as a 503 from the gateway. Warm requests are
# ~0.5-1s, so the fix is to not be cold.
#
# Five minutes is under Lambda's idle reclaim window without being chatty: ~8,600
# invocations a month at roughly half a second each, which is cents. Provisioned
# concurrency would solve it more thoroughly at about 120x the price, which is not a
# trade a demo should make.
#
# This buys the FIRST container only. A second simultaneous visitor still gets a cold
# one; the demo page retries on 503 for that case rather than pretending otherwise.
resource "aws_cloudwatch_event_rule" "warm" {
  name                = "${local.name}-warm"
  description         = "Keep one container warm so the first /ask does not exceed the gateway's 30s cap"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "warm" {
  rule  = aws_cloudwatch_event_rule.warm.name
  arn   = aws_lambda_function.app.arn
  input = jsonencode({ warm = true })

  # EventBridge invokes through a role rather than a Lambda resource policy, for the
  # same reason API Gateway does: resource policies are not enforced on this account.
  role_arn = aws_iam_role.warm_invoke.arn
}

resource "aws_iam_role" "warm_invoke" {
  name = "${local.name}-warm-invoke"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "warm_invoke" {
  name = "${local.name}-invoke"
  role = aws_iam_role.warm_invoke.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["lambda:InvokeFunction"]
      Resource = [aws_lambda_function.app.arn]
    }]
  })
}
