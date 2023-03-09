export const MapMode = {
  SHOW_POINTS: 'show_points',
  SELECT: 'select',
}

export const MapModeTooltip = {
  [MapMode.SHOW_POINTS]: '显示OD点',
  [MapMode.SELECT]: '选择OD簇'
}

export interface MapViewState {
  clusterLayerSvg: any,
  odLayerSvg: any,
  trjLayerSvg: any,
  clusterLayerShow: Boolean,
  codLayerShow: Boolean,
  trjLayeShow: Boolean,
}