output "demo_url" {
  description = "The public demo endpoint"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  description = "Push the image here before applying with a new tag"
  value       = aws_ecr_repository.app.repository_url
}

output "log_group" {
  value = aws_cloudwatch_log_group.lambda.name
}
