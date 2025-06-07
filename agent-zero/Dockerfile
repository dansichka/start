# Node 18 base image
FROM node:18-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /a0

COPY package*.json ./
RUN npm cache clean --force
RUN npm install

COPY . .

CMD ["node", "index.js"]
