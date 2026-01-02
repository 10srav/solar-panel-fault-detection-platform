export interface Panel {
  id: string
  name: string
  location: string | null
  installation_date: string | null
  capacity_kw: number | null
  created_at: string
  updated_at: string
}

export interface FaultEvent {
  id: string
  panel_id: string
  fault_class: string
  confidence: number
  severity_score: number
  risk_level: 'Low' | 'Medium' | 'High'
  fault_area_ratio: number | null
  temperature_score: number | null
  growth_rate: number | null
  alert_triggered: boolean
  alert_acknowledged: boolean
  detected_at: string
}

export interface PanelHistory {
  panel: Panel
  fault_events: FaultEvent[]
  total_events: number
  high_risk_events: number
  latest_severity: number | null
}

export interface ClassProbability {
  class_name: string
  probability: number
}

export interface Severity {
  fault_area_ratio: number
  temperature_score: number
  growth_rate: number
  severity_score: number
  risk_level: string
  alert_triggered: boolean
}

export interface RGBInferenceResponse {
  predicted_class: string
  class_index: number
  confidence: number
  all_probabilities: ClassProbability[]
  gradcam_overlay_base64: string | null
}

export interface ThermalInferenceResponse {
  fault_area_ratio: number
  mask_overlay_base64: string | null
}

export interface CombinedInferenceResponse {
  predicted_class: string
  class_index: number
  confidence: number
  all_probabilities: ClassProbability[]
  fault_area_ratio: number
  severity: Severity
  gradcam_overlay_base64: string | null
  mask_overlay_base64: string | null
  panel_id: string | null
  timestamp: string
}
