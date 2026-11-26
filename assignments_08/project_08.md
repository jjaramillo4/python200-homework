# Week 8 Project

## Part A: Supabase Setup

<!-- TODO: confirm both tables exist in the Table Editor, then write one sentence here. -->

## Part B: Cloud Cost Analysis

All costs are on-demand pricing in US East (N. Virginia), in USD per month. Scenario A is from the AWS Pricing Calculator. Scenario B is estimated from AWS's published hourly rates, so the calculator may differ slightly.

**Scenario A (lightweight compute):** A t3.micro EC2 instance running 160 hours per month costs about **$1.66/month**.

**Scenario B (heavy analytics workload):** About **$2,390/month** in total:
- EC2 p3.2xlarge (1 V100 GPU), 730 hours: about $2,234
- RDS for PostgreSQL db.m5.large, Single-AZ: about $132
- S3 Standard, 1 TB: about $24

Nothing surprised me, since with the cloud you are paying for usage.

**Comparison:** Scenario B costs about 1,400 times more than Scenario A, and almost all of that is the GPU instance. A GPU is worth it for a job that needs one, like training a model, run for a limited time and then shut down. Leaving one running 24/7 when it isn't doing GPU work wastes thousands of dollars a month.

## Video

<!-- TODO: paste video link -->
