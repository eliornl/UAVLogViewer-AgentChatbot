# UAV Log Viewer

![log seeking](preview.gif "Logo Title Text 1")

This is a Javascript based log viewer for Mavlink telemetry and dataflash logs.
[Live demo here](http://plot.ardupilot.org).

## Agentic Chatbot

Effortlessly analyze UAV telemetry with our intelligent AI chatbot. Ask natural questions like "How long was the total flight time?" or "Are there any anomalies in this flight?" and get instant, actionable insights.

For detailed setup instructions, API reference, and advanced usage guidance, refer to the [backend README](./backend/README.md).

## Build Setup

**Note:** If you want to use the AI chatbot functionality, start the backend first following the [backend README](./backend/README.md) instructions, then proceed with the frontend setup below.

If not using the backend:

```bash
git clone https://github.com/eliornl/UAVLogViewer-AgentChatbot.git
cd UAVLogViewer-AgentChatbot
```

### Initialize Submodules

Initialize and update the project submodules:

```bash
# Initialize submodules recursively
git submodule update --init --recursive
```

### Install and Run

``` bash
# install dependencies
npm install

# serve with hot reload at localhost:8080
npm run dev

# build for production with minification
npm run build

# run unit tests
npm run unit

# run e2e tests
npm run e2e

# run all tests
npm test
```