import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Upload, Camera, Thermometer, AlertCircle } from 'lucide-react'
import { inferRGB, inferCombined } from '../api/client'
import RiskBadge from '../components/RiskBadge'
import type { RGBInferenceResponse, CombinedInferenceResponse } from '../types'

export default function Inference() {
  const [rgbImage, setRgbImage] = useState<string | null>(null)
  const [thermalImage, setThermalImage] = useState<string | null>(null)
  const [result, setResult] = useState<RGBInferenceResponse | CombinedInferenceResponse | null>(null)

  const rgbMutation = useMutation({
    mutationFn: (imageBase64: string) => inferRGB(imageBase64),
    onSuccess: (data) => setResult(data),
  })

  const combinedMutation = useMutation({
    mutationFn: ({ rgb, thermal }: { rgb: string; thermal: string }) =>
      inferCombined(rgb, thermal),
    onSuccess: (data) => setResult(data),
  })

  const handleImageUpload = (
    event: React.ChangeEvent<HTMLInputElement>,
    type: 'rgb' | 'thermal'
  ) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onloadend = () => {
      const base64 = (reader.result as string).split(',')[1]
      if (type === 'rgb') {
        setRgbImage(base64)
      } else {
        setThermalImage(base64)
      }
    }
    reader.readAsDataURL(file)
  }

  const runInference = () => {
    if (rgbImage && thermalImage) {
      combinedMutation.mutate({ rgb: rgbImage, thermal: thermalImage })
    } else if (rgbImage) {
      rgbMutation.mutate(rgbImage)
    }
  }

  const isLoading = rgbMutation.isPending || combinedMutation.isPending
  const isCombinedResult = result && 'severity' in result

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">
          Run Fault Detection
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* RGB Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Camera className="inline h-4 w-4 mr-1" />
              RGB Image
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-400 transition-colors">
              {rgbImage ? (
                <div>
                  <img
                    src={`data:image/jpeg;base64,${rgbImage}`}
                    alt="RGB"
                    className="max-h-48 mx-auto rounded"
                  />
                  <button
                    onClick={() => setRgbImage(null)}
                    className="mt-2 text-sm text-red-600 hover:text-red-800"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                  <span className="text-sm text-gray-500">
                    Click to upload RGB image
                  </span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleImageUpload(e, 'rgb')}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          </div>

          {/* Thermal Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Thermometer className="inline h-4 w-4 mr-1" />
              Thermal Image (Optional)
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-400 transition-colors">
              {thermalImage ? (
                <div>
                  <img
                    src={`data:image/jpeg;base64,${thermalImage}`}
                    alt="Thermal"
                    className="max-h-48 mx-auto rounded"
                  />
                  <button
                    onClick={() => setThermalImage(null)}
                    className="mt-2 text-sm text-red-600 hover:text-red-800"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                  <span className="text-sm text-gray-500">
                    Click to upload thermal image
                  </span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleImageUpload(e, 'thermal')}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          </div>
        </div>

        <button
          onClick={runInference}
          disabled={!rgbImage || isLoading}
          className="mt-6 w-full py-3 px-4 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Results</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Classification Result */}
            <div className="border rounded-lg p-4">
              <h4 className="font-medium text-gray-700 mb-4">Classification</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Predicted Class</span>
                  <span className="font-semibold text-lg">
                    {result.predicted_class}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Confidence</span>
                  <span className="font-semibold">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Probability bars */}
              <div className="mt-4 space-y-2">
                {result.all_probabilities.map((p) => (
                  <div key={p.class_name}>
                    <div className="flex justify-between text-sm">
                      <span>{p.class_name}</span>
                      <span>{(p.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary-500 rounded-full"
                        style={{ width: `${p.probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Severity (if combined) */}
            {isCombinedResult && (
              <div className="border rounded-lg p-4">
                <h4 className="font-medium text-gray-700 mb-4">
                  Severity Assessment
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">Risk Level</span>
                    <RiskBadge
                      level={
                        (result as CombinedInferenceResponse).severity
                          .risk_level as any
                      }
                    />
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">Severity Score</span>
                    <span className="font-semibold">
                      {(
                        (result as CombinedInferenceResponse).severity
                          .severity_score * 100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">Fault Area</span>
                    <span>
                      {(
                        (result as CombinedInferenceResponse).fault_area_ratio *
                        100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>

                  {(result as CombinedInferenceResponse).severity
                    .alert_triggered && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-red-600" />
                      <span className="text-red-800 font-medium">
                        High Risk Alert Triggered
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Visualizations */}
          {(result.gradcam_overlay_base64 ||
            (isCombinedResult &&
              (result as CombinedInferenceResponse).mask_overlay_base64)) && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
              {result.gradcam_overlay_base64 && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">
                    Grad-CAM Visualization
                  </h4>
                  <img
                    src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                    alt="Grad-CAM"
                    className="w-full rounded-lg"
                  />
                  <p className="text-sm text-gray-500 mt-2">
                    Highlights regions contributing to the prediction
                  </p>
                </div>
              )}
              {isCombinedResult &&
                (result as CombinedInferenceResponse).mask_overlay_base64 && (
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">
                      Segmentation Mask
                    </h4>
                    <img
                      src={`data:image/png;base64,${
                        (result as CombinedInferenceResponse).mask_overlay_base64
                      }`}
                      alt="Segmentation"
                      className="w-full rounded-lg"
                    />
                    <p className="text-sm text-gray-500 mt-2">
                      Shows detected fault regions
                    </p>
                  </div>
                )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
