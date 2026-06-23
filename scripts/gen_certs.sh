#!/usr/bin/env bash
# Generate a self-signed TLS cert so the dashboard can be served over HTTPS.
# The browser webcam (getUserMedia, used by the Enroll page) only works on a
# secure context — https:// or http://localhost — so a LAN deployment served
# over plain http://10.x.x.x can't access the camera without this.
#
# Browsers will warn once about the self-signed cert (expected on a LAN); the
# SANs below cover localhost and this host's private IPs so the warning is the
# only friction. Re-run any time the host's IP changes.
set -euo pipefail

CERT_DIR="infrastructure/certs"
mkdir -p "$CERT_DIR"

sans="DNS:localhost,IP:127.0.0.1"
for ip in $(hostname -I); do
  case "$ip" in
    10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) sans="$sans,IP:$ip" ;;
  esac
done

echo "Generating self-signed cert with SANs: $sans"
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$CERT_DIR/server.key" -out "$CERT_DIR/server.crt" \
  -days 825 -subj "/CN=third-eye" -addext "subjectAltName=$sans" 2>/dev/null

chmod 600 "$CERT_DIR/server.key"
echo "Wrote $CERT_DIR/server.crt and $CERT_DIR/server.key"
echo "Now run 'make serve' (or 'make run'); it will serve HTTPS automatically."
