docker run -d -p 3000:3000 --name waha --restart unless-stopped devlikeapro/waha

```compose
version: "3.8"

services:
  waha:
    image: devlikeapro/waha
    container_name: waha
    restart: unless-stopped
    ports:
      - "3001:30001"
```

WAHA_API_KEY=6e24fa9b566e4c82b60a6af8f62f791f

WAHA_DASHBOARD_USERNAME=admin

WAHA_DASHBOARD_PASSWORD=bb2178930f214eab91a24a33a7f5325e

WHATSAPP_SWAGGER_USERNAME=admin

WHATSAPP_SWAGGER_PASSWORD=bb2178930f214eab91a24a33a7f5325e