import { useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, useMapEvents, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

interface LocationPickerMapProps {
  latitude: number
  longitude: number
  onChangeLocation: (lat: number, lng: number) => void
}

function MapClickHandler({ onChangeLocation }: { onChangeLocation: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onChangeLocation(Number(e.latlng.lat.toFixed(6)), Number(e.latlng.lng.toFixed(6)))
    },
  })
  return null
}

function MapRecenter({ lat, lng }: { lat: number; lng: number }) {
  const map = useMap()
  useEffect(() => {
    map.setView([lat, lng], map.getZoom())
  }, [lat, lng, map])
  return null
}

export default function LocationPickerMap({
  latitude,
  longitude,
  onChangeLocation,
}: LocationPickerMapProps) {
  const TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'

  return (
    <div className="relative w-full h-[320px] rounded-xl overflow-hidden border border-gray-700 bg-navy-900 shadow-inner">
      <MapContainer
        center={[latitude, longitude]}
        zoom={14}
        style={{ height: '100%', width: '100%', background: '#0b1329' }}
      >
        <TileLayer
          url={TILE_URL}
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        <MapClickHandler onChangeLocation={onChangeLocation} />
        <MapRecenter lat={latitude} lng={longitude} />

        {/* Selected Location Marker */}
        <CircleMarker
          center={[latitude, longitude]}
          radius={12}
          pathOptions={{
            color: '#3b82f6',
            fillColor: '#60a5fa',
            fillOpacity: 0.9,
            weight: 3,
          }}
        />
        <CircleMarker
          center={[latitude, longitude]}
          radius={22}
          pathOptions={{
            color: '#3b82f6',
            fillColor: '#3b82f6',
            fillOpacity: 0.2,
            weight: 1,
          }}
        />
      </MapContainer>

      {/* Helpful overlay badge */}
      <div className="absolute bottom-2.5 left-2.5 z-[1000] bg-navy-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-gray-700 text-[11px] text-gray-300 pointer-events-none">
        Click map anywhere to reposition marker
      </div>
    </div>
  )
}
