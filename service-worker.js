self.addEventListener("install", event => {
  event.waitUntil(
    caches.open("chatbot-cache").then(cache => {
      return cache.addAll([
        "/",
        "/index.html",
        "/static/css/style.css",
        "/static/js/app.js"
      ]);
    })
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});