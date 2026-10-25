variable "region" {
  description = "EU region, for data residency"
  type        = string
  default     = "eu-central-1"
}

variable "image_tag" {
  description = "Image tag to deploy. Set to the git SHA so a deploy is traceable to a commit."
  type        = string
  default     = "latest"
}

variable "bedrock_model_id" {
  description = "Inference profile ID used for generation"
  type        = string
  default     = "eu.amazon.nova-lite-v1:0"
}

variable "bedrock_foundation_model" {
  description = "Underlying foundation model the profile routes to (needed for the IAM resource ARN)"
  type        = string
  default     = "amazon.nova-lite-v1:0"
}

variable "max_concurrency" {
  description = "Reserved concurrent executions, or -1 for no reservation. The hard ceiling on simultaneous Bedrock calls."
  type        = number

  # -1, not 2, because reserving concurrency requires the account to keep at least
  # 10 executions unreserved. A new account's total limit IS 10, so any reservation
  # at all is rejected: "decreases account's UnreservedConcurrentExecution below its
  # minimum value of [10]". The account limit then serves as the ceiling instead,
  # which is the same order of magnitude as the cap this was meant to impose.
  # Set this to a real number once the account limit is raised above 10.
  default = -1
}

variable "rate_limit_per_min" {
  description = "Per-IP requests per minute, enforced in the application"
  type        = number
  default     = 10
}

variable "monthly_budget_eur" {
  description = "Budget alert threshold"
  type        = string
  default     = "5"
}

variable "alert_email" {
  description = "Where budget alerts go"
  type        = string
}
