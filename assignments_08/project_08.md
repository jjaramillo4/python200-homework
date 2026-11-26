# Week 8 Project

## Part A: Supabase Setup

<!-- TODO: confirm both tables exist in the Table Editor, then write one sentence here. -->

## Part B: Cloud Cost Analysis

All costs are from the AWS Pricing Calculator, using on-demand pricing in US East (N. Virginia), in USD per month.

**Scenario A (lightweight compute):** A t3.micro EC2 instance running 160 hours per month costs about **$1.66/month**.

**Scenario B (heavy analytics workload):** **$2,438.94/month** in total:
- EC2 p3.2xlarge (1 V100 GPU), 730 hours: $2,233.80
- RDS for PostgreSQL db.m5.large, Single-AZ, 100 GB gp2 storage: $181.59
- S3 Standard, 1 TB: $23.55

Nothing surprised me, since with the cloud you are paying for usage.

**Comparison:** Scenario B costs about 1,500 times more than Scenario A, and almost all of that is the GPU instance. A GPU is worth it for a job that needs one, like training a model, run for a limited time and then shut down. Leaving one running 24/7 when it isn't doing GPU work wastes thousands of dollars a month.

## Video

<!-- TODO: paste video link -->
