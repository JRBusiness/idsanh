vault {
  renew_token = false
  vault_agent_token_file = "/home/vault/.vault-token"
  retry {
    backoff = "1s"
  }
}

template {
  destination = "/etc/secrets/env.txt"
  contents = <<EOH
{{- with secret "secret/GLOBAL/ID_SCANNER" }}
SENTRY_DSN={{ .Data.data.SENTRY_DSN }}
{{ end }}


{{- with secret "secret/PRODUCTION/GLOBAL" }}
DEBUG={{ .Data.data.DEBUG }}
SENTRY_ENVIRONMENT="{{ .Data.data.SENTRY_ENVIRONMENT }}"
{{ end }}

{{- with secret "secret/PRODUCTION/ID_SCANNER" }}
APP_NAME={{ .Data.data.APP_NAME }}
APP_VERSION={{ .Data.data.APP_VERSION }}
FLASK_RUN_PORT={{ .Data.data.FLASK_RUN_PORT }}
FLASK_RUN_HOST={{ .Data.data.FLASK_RUN_HOST }}
FLASK_DEBUG={{ .Data.data.FLASK_DEBUG }}
API_KEY={{ .Data.data.API_KEY }}
{{ end }}

EOH
}

