use std::sync::Arc;
use tokio::sync::Mutex;
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct EmbeddingCache {
    cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
}

impl EmbeddingCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1000).unwrap())
            ))),
        }
    }

    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.cache.lock().await.get(key).cloned()
    }

    pub async fn set(&self, key: String, value: Vec<f32>) {
        self.cache.lock().await.put(key, value);
    }
}
