export interface GlobalState {
  pointsExist: Boolean,
  timeScope: [number, number],
  dateScope: [number, number],
  odPoints: Array<[]>,
  odIndexList: number[],
  pointClusterMap: Map<number, number>,
  clusterPointMap: Map<number, number[]>,
}