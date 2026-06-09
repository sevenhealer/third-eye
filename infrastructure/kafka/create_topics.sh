#!/bin/bash
set -e

BOOTSTRAP="localhost:9092"
echo "[kafka-init] Waiting for Kafka to be ready..."
sleep 10

create_topic() {
    local name=$1
    local partitions=$2
    local retention_ms=$3

    kafka-topics.sh \
        --bootstrap-server "$BOOTSTRAP" \
        --create \
        --if-not-exists \
        --topic "$name" \
        --partitions "$partitions" \
        --replication-factor 1 \
        --config retention.ms="$retention_ms" \
        --config cleanup.policy=delete

    echo "[kafka-init] Created topic: $name (partitions=$partitions, retention=${retention_ms}ms)"
}

# camera.frames — raw frame metadata (not pixel data), 1 partition per camera
create_topic "camera.frames"    8   60000         # 1 minute

# camera.health — tamper and disconnect events
create_topic "camera.health"    2   86400000      # 1 day

# Face pipeline
create_topic "detections.faces"         4   3600000   # 1 hour
create_topic "antispoofing.results"     4   3600000   # 1 hour
create_topic "recognition.identity"     4   86400000  # 1 day
create_topic "enrollment.candidates"    2   604800000 # 7 days

# Object and tracking
create_topic "detections.objects"   4   3600000   # 1 hour
create_topic "tracking.states"      4   3600000   # 1 hour
create_topic "counts.snapshot"      4   3600000   # 1 hour

# Behavioral
create_topic "events.actions"   4   86400000  # 1 day
create_topic "scene.context"    4   3600000   # 1 hour

# Events and alerts
create_topic "events.detected"  2   604800000  # 7 days
create_topic "alerts.outbound"  2   604800000  # 7 days

# Feedback and retraining
create_topic "feedback.corrections" 2   2592000000  # 30 days

echo "[kafka-init] All topics created successfully."
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list
