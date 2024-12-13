#!/bin/bash

celery -A celery_worker:celery worker --loglevel=INFO --concurrency=$1 --logfile outputs/celery.logs
