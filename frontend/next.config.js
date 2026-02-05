/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // API proxy to backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.API_URL || 'http://localhost:8000/api/:path*',
      },
    ];
  },
  
  // Environment variables exposed to browser
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
  },
  
  // Image optimization
  images: {
    domains: ['localhost', 'minio', 'adclass.ai'],
    formats: ['image/avif', 'image/webp'],
  },
  
  // Experimental features
  experimental: {
    serverActions: true,
  },
};

module.exports = nextConfig;
