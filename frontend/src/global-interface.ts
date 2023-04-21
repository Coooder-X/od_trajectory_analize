export interface GlobalState {
  pointsExist: Boolean,
  timeScope: [number, number],
  dateScope: [number, number],
  odPoints: Array<[]>,
  odIndexList: number[],
  pointClusterMap: Map<number, number>,
  clusterPointMap: Map<number, number[]>,
  adjTable: Map<number, number[]>,
  forceTreeLinks: ForceLink,
  forceTreeNodes: ForceNode,
}

export type ForceLink = Array<{source: number, target: number, isFake?: boolean}>;
export type ForceNode = Array<{name: string}>;