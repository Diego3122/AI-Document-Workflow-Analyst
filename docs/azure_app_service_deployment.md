# Azure App Service Deployment

This project is designed to deploy as a single Linux custom container on Azure App Service.

## Runtime shape

- Streamlit is the public UI on port `8501`
- FastAPI runs privately on `127.0.0.1:8000` inside the same container
- Microsoft Entra sign-in is enforced by Azure App Service Authentication
- FastAPI still validates the forwarded JWT server-side
- uploads, SQLite, logs, and Chroma data persist under `/home/site/data`

## Recommended Azure services

- Azure App Service for Containers
- Azure Container Registry
- Microsoft Entra app registration for tenant-only sign-in
- Azure Key Vault or App Service settings for secrets

## Required app settings

Start from `production.env.example` and configure at least:

- `ENVIRONMENT=production`
- `AUTH_MODE=jwt`
- `UI_PUBLIC_ENABLED=true`
- `JWT_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0`
- `JWT_AUDIENCES=api://<your-api-app-id-or-client-id>`
- `JWT_ALGORITHMS=RS256`
- `JWT_DEFAULT_ROLE=analyst`
- `JWT_ADMIN_EMAILS=you@yourtenant.com`
- `TRUSTED_HOSTS=<your-app-name>.azurewebsites.net,127.0.0.1,localhost`
- `CORS_ALLOWED_ORIGINS=https://<your-app-name>.azurewebsites.net`
- `UI_FORWARDED_ACCESS_TOKEN_HEADERS=X-Ms-Token-Aad-Access-Token,Authorization`
- `TRUST_PROXY_HEADERS=true`
- `PROXY_TRUSTED_IPS=*`
- `WEBSITES_PORT=8501`
- `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`
- `UPLOADS_DIR=/home/site/data/uploads`
- `CHROMA_DIR=/home/site/data/chroma`
- `LOGS_DIR=/home/site/data/logs`
- `SQLITE_PATH=/home/site/data/logs/app.db`
- `DOCUMENT_RETENTION_DAYS=30`
- `GEMINI_API_KEY=<secret>`
- `LLM_PROVIDER=auto`
- `EMBEDDING_PROVIDER=auto`

## Authentication notes

- Keep Azure App Service Authentication enabled and require sign-in.
- For the first deployment, use tenant-only Microsoft Entra access.
- Keep the UI-to-API hop on `http://127.0.0.1:8000`; HTTPS is only required on the public App Service edge.
- The app grants basic upload/query capability to authenticated JWT users through `JWT_DEFAULT_ROLE=analyst`.
- Use `JWT_ADMIN_EMAILS` or `JWT_ADMIN_SUBJECTS` to keep metrics, review tools, and admin actions restricted to your account.

## Deployment flow

1. Create an Azure Container Registry.
2. Build and push the image from this repo.
3. Create an App Service for Containers using the pushed image.
4. Enable system-assigned identity on the App Service.
5. Grant the App Service identity pull access to ACR.
6. Configure App Service Authentication with Microsoft Entra.
7. Add the production app settings from `production.env.example`.
8. Restart the App Service and verify the UI loads after sign-in.

## Verification checklist

- `/livez` responds successfully from the container
- the Streamlit UI loads through App Service on port `8501`
- sign-in is required before the dashboard appears
- a signed-in user can upload, prepare, query, and delete their own file
- admin-only metrics and review tools remain hidden for standard users
- data persists across restart and purges after the retention window
