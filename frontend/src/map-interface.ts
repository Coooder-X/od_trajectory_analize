export const MapMode = {
  SHOW_POINTS: 'show_points',
  HIDE_POINTS: 'hide_points',
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
}