const CACHE_NAME = "chatbot-cache-v1";
const urlsToCache = [
  "/static/images/bot.png",
  "/static/images/bot.png",
  "/static/manifest.json"
];

// Install event - cache files
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
      .catch(err => console.log("Cache failed:", err))
  );
});

// Activate event - cleanup old caches
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key !== CACHE_NAME)
            .map(key => caches.delete(key))
      );
    })
  );
});

// Fetch event - serve cached files if available
self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});