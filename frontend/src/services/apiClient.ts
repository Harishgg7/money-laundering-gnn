/**
 * apiClient.ts — Core HTTP Client and Mock Fallback Engine.
 * 
 * Separates UI components from raw fetch calls. When the backend REST endpoints are 
 * available, queries hit `/api/*`. In Phase 1 or during offline development,
 * clean mock fixtures are served with realistic asynchronous latency simulation.
 */

// In Phase 2, TRACE communicates directly with the Python backend REST API.
// Mock fallback is disabled by default to ensure all presented data is real.
const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';

export async function request<T>(endpoint: string, options?: RequestInit, mockFallback?: () => Promise<T> | T): Promise<T> {
  if (USE_MOCK && mockFallback) {
    await new Promise((resolve) => setTimeout(resolve, 150));
    return mockFallback();
  }

  try {
    const res = await fetch(endpoint, {
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      ...options,
    });

    if (!res.ok) {
      let errDetails = res.statusText;
      try {
        const errJson = await res.json();
        if (errJson && errJson.error) {
          errDetails = errJson.error;
        }
      } catch {
        // use default statusText
      }
      throw new Error(`API Error [${res.status}]: ${errDetails}`);
    }

    return (await res.json()) as T;
  } catch (err) {
    // In Phase 2, do NOT silently mask network errors with mock data in production-facing flow.
    if (USE_MOCK && mockFallback) {
      console.warn(`[TRACE API] Development fallback for ${endpoint}:`, err);
      return mockFallback();
    }
    throw err;
  }
}
