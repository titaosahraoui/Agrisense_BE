import ee
import os

def init_ee():
    try:
        ee.Initialize()
        print("✅ Earth Engine initialized")
    except Exception as e:
        print("⚠️ EE not initialized, authenticating...")
        ee.Authenticate()
        ee.Initialize()
        print("✅ Earth Engine authenticated & initialized")
