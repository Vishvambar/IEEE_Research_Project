#!/bin/bash
echo "============================================="
echo "   CLINIVORA MASTER STARTUP SCRIPT"
echo "============================================="
echo "Starting FastAPI Backend Server..."

# Start FastAPI backend in the background
python api/main.py &
BACKEND_PID=$!

echo "Backend started on PID $BACKEND_PID"
echo "Starting React Frontend Server..."

# Start Vite frontend
cd clinivora-app
npm run dev &
FRONTEND_PID=$!

echo "Frontend started on PID $FRONTEND_PID"
echo "============================================="
echo "Both servers are running! "
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:5173"
echo "Press [CTRL+C] to stop both servers."
echo "============================================="

# Wait for user interrupt
trap "echo 'Stopping servers...'; kill $BACKEND_PID; kill $FRONTEND_PID; exit" INT
wait
