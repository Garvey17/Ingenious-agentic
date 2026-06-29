/**
 * API Client for interacting with the Ingenious Agentic backend.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

/**
 * Helper to handle fetch requests and throw structured errors
 */
async function fetchApi(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    const headers = {
        'Content-Type': 'application/json',
        ...options.headers,
    };

    try {
        const response = await fetch(url, {
            ...options,
            headers,
        });

        if (!response.ok) {
            let errorMessage = `HTTP Error ${response.status}`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
            } catch (e) {
                // Ignored
            }
            throw new Error(errorMessage);
        }

        return await response.json();
    } catch (error) {
        console.error(`API Error [${options.method || 'GET'} ${endpoint}]:`, error);
        throw error;
    }
}

/**
 * Start a new research task
 * @param {string} topic - The research topic
 * @param {string} depth - 'quick', 'standard', or 'deep'
 * @param {number} maxSources - Max sources to gather (default 10)
 */
export async function startResearch(topic, depth = 'standard', maxSources = 10) {
    return fetchApi('/research/', {
        method: 'POST',
        body: JSON.stringify({
            topic,
            depth,
            max_sources: maxSources,
        }),
    });
}

/**
 * Get the current status of a research task
 * @param {string} request_id - The UUID of the task
 */
export async function getResearchStatus(request_id) {
    return fetchApi(`/research/${request_id}`);
}

/**
 * Get the granular state of the LangGraph execution for an active task
 * @param {string} request_id - The UUID of the task
 */
export async function getResearchState(request_id) {
    return fetchApi(`/research/${request_id}/state`);
}

/**
 * Check the health and configuration of the backend
 */
export async function checkSystemHealth() {
    try {
        const healthUrl = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api').replace('/api', '/health');
        const response = await fetch(healthUrl);
        if (!response.ok) return { healthy: false };
        return await response.json();
    } catch (error) {
        return { healthy: false, error: error.message };
    }
}

/**
 * Fetch the number of items stored in Qdrant memory
 */
export async function getMemoryCount() {
    return fetchApi('/memory/count');
}

/**
 * Fetch the list of dynamically discovered MCP tools
 */
export async function getMcpTools() {
    return fetchApi('/mcp/tools');
}
