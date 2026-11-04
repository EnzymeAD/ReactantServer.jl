#!/usr/bin/env bash
#
# Start Prometheus + Grafana pointed at a ReactantServer node's metrics, with podman or docker and
# no compose tool.
#
#   monitor.sh up [TARGET] [options]   start (or restart) a stack scraping TARGET
#   monitor.sh down [--name N]         stop and remove it (its stored data is kept)
#   monitor.sh purge [--name N]        stop it and delete its stored data
#   monitor.sh status [--name N]       its containers and whether the scrape works
#   monitor.sh logs [--name N]         follow Prometheus' and Grafana's logs
#
# TARGET is where the node's metrics port answers (the default is 8002 on this host):
#   8002                    a port published on this host, such as a node container or pod's
#   netai03:8002            any host:port reachable from a container
#   reactantserver:8002     a container or pod on a container network: add --network NAME
#
# Options:
#   --name N                stack name, so several can run at once (default: rs-monitoring)
#   --network NAME          join this existing container network, to reach a target by its name
#   --grafana-port P        host port for Grafana (default 3000)
#   --prometheus-port P     host port for Prometheus (default 9090)
#
# The engine is podman when installed, else docker; ENGINE=docker forces one. The dashboards are
# mounted from grafana/dashboards beside this script, so regenerating them (grafana/build.jl) shows
# up in Grafana within 30 seconds, without a restart.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMETHEUS_IMAGE="docker.io/prom/prometheus:latest"
GRAFANA_IMAGE="docker.io/grafana/grafana:latest"

die() { echo "monitor.sh: $*" >&2; exit 1; }

ENGINE="${ENGINE:-$(command -v podman >/dev/null 2>&1 && echo podman || echo docker)}"
command -v "$ENGINE" >/dev/null 2>&1 || die "neither podman nor docker is installed"

cmd="${1:-}"
[ -n "$cmd" ] && shift || true
target=""
name="rs-monitoring"
network=""
gport=3000
pport=9090
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --network) network="$2"; shift 2 ;;
    --grafana-port) gport="$2"; shift 2 ;;
    --prometheus-port) pport="$2"; shift 2 ;;
    -h|--help) sed -n '3,25p' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) [ -z "$target" ] || die "more than one target given"; target="$1"; shift ;;
  esac
done

state="${XDG_STATE_HOME:-$HOME/.local/state}/reactantserver-monitoring/$name"
prom="$name-prometheus"
graf="$name-grafana"
own_net="$name-net"

# The host as seen from a container: podman names it host.containers.internal itself; docker needs
# the name mapped to its host gateway.
host_alias() { [ "$(basename "$ENGINE")" = podman ] && echo host.containers.internal || echo host.docker.internal; }

resolve_target() {
  local t="${1:-8002}"
  t="${t#http://}"; t="${t%/metrics}"; t="${t%/}"
  case "$t" in
    :*) t="${t#:}" ;;
  esac
  case "$t" in
    *[!0-9]*) ;;                        # host:port (or a bare name, which gets :8002)
    *) t="$(host_alias):$t" ;;          # a bare port: published on this host
  esac
  case "$t" in
    *:*) echo "$t" ;;
    *) echo "$t:8002" ;;
  esac
}

remove_containers() { "$ENGINE" rm -f "$prom" "$graf" >/dev/null 2>&1 || true; }

up() {
  local t net extra=()
  t="$(resolve_target "$target")"
  mkdir -p "$state/datasources"

  cat >"$state/prometheus.yml" <<EOF
# Written by monitor.sh; rerun it to change the target.
global:
  scrape_interval: 15s
  scrape_timeout: 10s
scrape_configs:
  - job_name: reactantserver
    metrics_path: /metrics
    static_configs:
      - targets: ["$t"]
  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
EOF
  cat >"$state/datasources/datasource.yml" <<EOF
# Written by monitor.sh.
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://$prom:9090
    isDefault: true
    jsonData:
      timeInterval: 15s
      httpMethod: POST
    editable: true
EOF

  if [ -n "$network" ]; then
    "$ENGINE" network inspect "$network" >/dev/null 2>&1 || die "no container network named $network ($ENGINE network ls lists them)"
    net="$network"
  else
    net="$own_net"
    "$ENGINE" network inspect "$net" >/dev/null 2>&1 || "$ENGINE" network create "$net" >/dev/null
  fi
  [ "$(basename "$ENGINE")" = podman ] || extra+=(--add-host "host.docker.internal:host-gateway")

  remove_containers
  "$ENGINE" run -d --name "$prom" --network "$net" ${extra[@]+"${extra[@]}"} --restart unless-stopped \
    -p "$pport:9090" \
    -v "$state/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
    -v "$name-prometheus-data:/prometheus" \
    "$PROMETHEUS_IMAGE" \
    --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time=15d --web.enable-lifecycle >/dev/null
  "$ENGINE" run -d --name "$graf" --network "$net" --restart unless-stopped \
    -p "$gport:3000" \
    -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    -e GF_SECURITY_ADMIN_USER=admin -e "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}" \
    -e GF_USERS_DEFAULT_THEME=dark \
    -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/overview.json \
    -v "$state/datasources:/etc/grafana/provisioning/datasources:ro" \
    -v "$HERE/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro" \
    -v "$HERE/grafana/dashboards:/var/lib/grafana/dashboards:ro" \
    -v "$name-grafana-data:/var/lib/grafana" \
    "$GRAFANA_IMAGE" >/dev/null

  echo "scraping $t (network $net)"
  echo "Grafana:    http://$(hostname -f 2>/dev/null || hostname):$gport"
  echo "Prometheus: http://$(hostname -f 2>/dev/null || hostname):$pport"
  wait_for_scrape
}

# Ask Prometheus whether the reactantserver target answers, for up to a minute.
target_health() {
  curl -s -m 5 "http://localhost:$pport/api/v1/targets?scrapePool=reactantserver" 2>/dev/null || true
}
wait_for_scrape() {
  command -v curl >/dev/null 2>&1 || { echo "(install curl to have the scrape checked)"; return 0; }
  local body
  for _ in $(seq 1 30); do
    body="$(target_health)"
    case "$body" in
      *'"health":"up"'*) echo "scrape: up"; return 0 ;;
      *'"health":"down"'*)
        echo "scrape: DOWN: $(printf '%s' "$body" | sed -n 's/.*"lastError":"\([^"]*\)".*/\1/p')"
        echo "  Check that the target answers from a container. A node on this host publishes its port"
        echo "  on $(host_alias); a container on a named network needs --network and its name."
        return 1 ;;
    esac
    sleep 2
  done
  echo "scrape: no result yet; check with: $0 status --name $name"
}

status() {
  "$ENGINE" ps -a --filter "name=^$name-" --format '{{.Names}}  {{.Status}}  {{.Ports}}'
  [ -f "$state/prometheus.yml" ] && echo "target: $(sed -n 's/.*targets: \["\([^"]*\)"\].*/\1/p' "$state/prometheus.yml" | head -1)"
  command -v curl >/dev/null 2>&1 || return 0
  local body
  body="$(target_health)"
  case "$body" in
    *'"health":"up"'*) echo "scrape: up" ;;
    *'"health":"down"'*) echo "scrape: DOWN: $(printf '%s' "$body" | sed -n 's/.*"lastError":"\([^"]*\)".*/\1/p')" ;;
    *) echo "scrape: Prometheus not answering on localhost:$pport" ;;
  esac
}

down() {
  remove_containers
  "$ENGINE" network rm "$own_net" >/dev/null 2>&1 || true
  echo "stopped $name (data kept; purge deletes it)"
}

case "$cmd" in
  up) up ;;
  down) down ;;
  purge) down; "$ENGINE" volume rm -f "$name-prometheus-data" "$name-grafana-data" >/dev/null 2>&1 || true; rm -rf "$state"; echo "deleted $name's data" ;;
  status) status ;;
  logs) "$ENGINE" logs -f "$prom" & "$ENGINE" logs -f "$graf"; wait ;;
  ""|-h|--help) sed -n '3,25p' "$0" ;;
  *) die "unknown command $cmd (up, down, purge, status, logs)" ;;
esac
