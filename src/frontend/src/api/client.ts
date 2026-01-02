import axios from 'axios'
import type {
  Panel,
  PanelHistory,
  RGBInferenceResponse,
  ThermalInferenceResponse,
  CombinedInferenceResponse
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Panel APIs
export const fetchPanels = async (): Promise<Panel[]> => {
  const response = await client.get('/panels')
  return response.data
}

export const fetchPanel = async (panelId: string): Promise<Panel> => {
  const response = await client.get(`/panels/${panelId}`)
  return response.data
}

export const fetchPanelHistory = async (panelId: string): Promise<PanelHistory> => {
  const response = await client.get(`/panels/${panelId}/history`)
  return response.data
}

export const createPanel = async (data: Partial<Panel>): Promise<Panel> => {
  const response = await client.post('/panels', data)
  return response.data
}

// Inference APIs
export const inferRGB = async (
  imageBase64: string,
  generateGradcam = true
): Promise<RGBInferenceResponse> => {
  const response = await client.post('/infer/rgb', {
    image_base64: imageBase64,
    generate_gradcam: generateGradcam,
  })
  return response.data
}

export const inferThermal = async (
  imageBase64: string,
  threshold = 0.5
): Promise<ThermalInferenceResponse> => {
  const response = await client.post('/infer/thermal', {
    image_base64: imageBase64,
    threshold,
  })
  return response.data
}

export const inferCombined = async (
  rgbImageBase64: string,
  thermalImageBase64: string,
  panelId?: string
): Promise<CombinedInferenceResponse> => {
  const response = await client.post('/infer/combined', {
    rgb_image_base64: rgbImageBase64,
    thermal_image_base64: thermalImageBase64,
    panel_id: panelId,
  })
  return response.data
}

// Health check
export const checkHealth = async (): Promise<{
  status: string
  models_loaded: Record<string, boolean>
}> => {
  const response = await client.get('/health')
  return response.data
}

export default client
