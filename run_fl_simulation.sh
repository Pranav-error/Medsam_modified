#!/bin/bash
# Federated Learning Simulation Script
# Launches server + multiple hospital clients for testing

set -e

echo "🏥 Starting Federated Learning Simulation"
echo "=========================================="

# Configuration
NUM_CLIENTS=3
NUM_ROUNDS=5
SERVER_ADDRESS="127.0.0.1:8080"
DATA_DIR="data/breast_cancer"
MODEL_CHECKPOINT="work_dir/MedSAM/medsam_vit_b.pth"

# Check if model checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "⚠️  Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "Please download or train the MedSAM model first"
    exit 1
fi

# Create log directory
mkdir -p fl_logs

echo "📊 Configuration:"
echo "  - Number of hospital clients: $NUM_CLIENTS"
echo "  - Number of FL rounds: $NUM_ROUNDS"
echo "  - Server address: $SERVER_ADDRESS"
echo ""

# Start the server in the background
echo "🚀 Starting FL Server..."
python fl_server.py \
    --rounds $NUM_ROUNDS \
    --clients $NUM_CLIENTS \
    --min-clients 2 \
    --address $SERVER_ADDRESS \
    > fl_logs/server.log 2>&1 &

SERVER_PID=$!
echo "✓ Server started (PID: $SERVER_PID)"

# Wait for server to start
sleep 3

# Start clients in the background
for i in $(seq 0 $(($NUM_CLIENTS - 1))); do
    echo "🏥 Starting Hospital Client $i..."
    python fl_client.py \
        --client-id $i \
        --server $SERVER_ADDRESS \
        --data-dir $DATA_DIR \
        --checkpoint $MODEL_CHECKPOINT \
        > fl_logs/client_$i.log 2>&1 &
    
    CLIENT_PIDS[$i]=$!
    echo "✓ Client $i started (PID: ${CLIENT_PIDS[$i]})"
    sleep 1
done

echo ""
echo "✅ All clients connected!"
echo "📈 Federated learning in progress..."
echo "📋 Check logs in fl_logs/ directory"
echo ""
echo "To monitor progress:"
echo "  - Server log: tail -f fl_logs/server.log"
echo "  - Client 0 log: tail -f fl_logs/client_0.log"
echo ""

# Wait for server to complete
wait $SERVER_PID

echo ""
echo "🎉 Federated Learning Complete!"
echo "📁 Global model checkpoints saved in fl_checkpoints/"
echo ""
echo "To view FLED metrics, check the server log:"
echo "  grep -E 'fairness|accuracy|dice' fl_logs/server.log"
