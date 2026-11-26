# Week 8 Warmup

## Part 1: Cloud Concepts

### Question 1

The core economic model is that you are able to rent computing resources like storage without having to invest up front in physical hardware such as new servers, and you pay and scale based on usage.

### Question 2

The difference is that vertical scaling means upgrading your current setup, whereas horizontal scaling means adding more computers as needed instead of relying on one.

- **Vertical example:** When my desktop needed more power, I upgraded its RAM, storage, and CPU.
- **Horizontal example:** Our company server is overseas, so the connection is slow for us. Instead, you could add more servers in regions closer to each office so users connect to the nearest one.

**Scenarios:**

1. **Viral web app:** Horizontal, because 100,000 users is too much traffic for one server, so you spread multiple users across more servers.
2. **Slow model training:** Vertical, because the job runs on one machine that needs to be upgraded with a faster GPU and more RAM.
3. **10,000-file pipeline:** Horizontal, same idea of using multiple machines, since the files can be split up and processed at the same time.

### Question 3

**Classification:**

1. **Gmail:** SaaS, it's a finished app you just use.
2. **Azure Virtual Machines:** IaaS, you rent raw virtual machines and manage everything on them.
3. **AWS S3:** IaaS, it's raw storage you rent and organize yourself.
4. **GitHub Codespaces:** PaaS, it gives you a ready development environment and you bring your code.
5. **Snowflake:** PaaS, it's a managed data platform where you bring your data and queries and they run the database.
6. **Supabase:** BaaS, it's a ready-made backend (database, auth, API) that an app plugs into.

**Definitions:**

- **IaaS:** You rent raw computing resources like virtual machines or storage. Example: Azure Virtual Machines. I'm responsible for the operating system, software, updates, and security on the machine.
- **PaaS:** The provider runs the servers and operating system, and I bring my code and data. Example: GitHub Codespaces. I'm responsible for my code, data, and settings.
- **SaaS:** A finished app that I just use. Example: Gmail. I'm only responsible for my account and how I use it.

### Question 4

With AWS you rent servers, databases, and storage, but you configure and connect them yourself. Snowflake and Databricks are data platforms that are already built and working, so you just load the data. You gain convenience with these platforms, but you lose control, there's probably an additional cost, and it's harder to switch away later.

### Question 5

If you can run your database locally or compute the work locally as well, because your data fits on one machine and you don't need massive compute, then local is often faster and cheaper, especially for an early prototype.

## Part 2: Cloud Landscape

### Question 1

- **AWS:** The oldest and largest provider with the broadest range of services, used by large enterprises, startups, and nonprofits with engineering staff.
- **GCP:** Strongest in data and machine learning, used by teams doing large-scale analytics or ML work.
- **Azure:** Built around Microsoft tools like Windows, Active Directory, and Microsoft 365, used by enterprises and government organizations that already run on Microsoft.

### Question 2

1. **Access:** Azure needed organizational setup and invitations that could block students for days, while a Supabase account takes a couple of minutes and the free tier covers the course.
2. **Pedagogical fit:** Supabase stores data as tables with rows and columns, which is a more transferable skill than Azure Blob Storage's file-based storage.
3. **Pipeline coherence:** The raw and enriched stages of the pipeline map to two related tables, which makes each stage easy to inspect and debug.

**Reflection:** When choosing a cloud tool, I look at how well it integrates with my current setup, whether I can use it later with other software, and its convenience and ease of use, which AWS does not provide.

### Question 3

1. **10 TB of images retrieved by filename:** Object storage, AWS S3.
2. **GPU training job for four hours:** ML platform, AWS SageMaker.
3. **Web API that scales with traffic:** Serverless compute, AWS Lambda.
4. **Send data to an LLM and get text back:** LLM API, Azure OpenAI.

### Question 4

**Project:** Scrape game data as new information is released, process it and calculate results on Heroku, and store it in a Supabase database using SQL from Python.

**Stack:**
- **Heroku** (PaaS): runs the scraper and processing code.
- **Supabase** (managed relational DB): stores the scraped data and results.

**One provider?** It doesn't really matter much for me, since most of my projects don't need the cloud. Consolidating would give one bill and one place to manage everything, but you give up the flexibility of picking the best tool for each job.
