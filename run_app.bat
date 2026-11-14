@echo off
title RetinaHeart AI - Retinal Image Analysis for Heart Disease Risk Prediction
echo =========================================================================
echo  Starting RetinaHeart AI Web Application Server...
echo  Title: Retinal image analysis for heart disease risk prediction
echo  Open Browser: http://127.0.0.1:5000
echo  Admin Credentials: admin / admin
echo =========================================================================
set KMP_DUPLICATE_LIB_OK=TRUE
"C:\Users\dell\anaconda3\python.exe" app.py
pause
