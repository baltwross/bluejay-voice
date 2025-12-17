# =============================================================================
# Bluejay Terminator - Frontend Dockerfile
# =============================================================================
# Multi-stage build supporting both development and production
#
# Stages:
#   - base: Common Node.js setup
#   - development: Vite dev server with hot reloading
#   - build: Production build step
#   - production: Nginx serving static files
#
# Usage:
#   Development: docker build --target development -t bluejay-frontend:dev .
#   Production:  docker build --target production -t bluejay-frontend:prod .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base - Common Node.js setup
# -----------------------------------------------------------------------------
FROM node:20-alpine AS base

WORKDIR /app

# Copy package files for dependency installation
COPY package*.json ./

# Install dependencies
RUN npm ci

# -----------------------------------------------------------------------------
# Stage 2: Development - Vite dev server with hot reloading
# -----------------------------------------------------------------------------
FROM base AS development

# Copy source files
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Start Vite dev server with host flag for container access
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# -----------------------------------------------------------------------------
# Stage 3: Build - Production build step
# -----------------------------------------------------------------------------
FROM base AS build

# Copy source files
COPY . .

# Build arguments for environment variables
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

# Build the application
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 4: Production - Nginx serving static files
# -----------------------------------------------------------------------------
FROM nginx:alpine AS production

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy public assets (fonts, icons)
COPY --from=build /app/public /usr/share/nginx/html/public

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

