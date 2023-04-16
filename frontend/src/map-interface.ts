export const MapMode = {
  INIT: 'init',
  ORIGIN_POINTS: 'origin_points',
  // SHOW_POINTS: 'show_points',
  // HIDE_POINTS: 'hide_points',
  CLUSTERED: 'clustered',
  SELECT: 'select',
}

export interface MapViewState {
  clusterLayerSvg: any,
  odLayerSvg: any,
  trjLayerSvg: any,
  clusterLayerShow: Boolean,
  codLayerShow: Boolean,
  trjLayeShow: Boolean,
  mapMode: Set<string>
}