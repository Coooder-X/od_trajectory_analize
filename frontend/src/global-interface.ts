export interface GlobalState {
  pointsExist: Boolean,
  timeScope: [number, number],
  dateScope: [number, number],
  odPoints: Array<[]>,
  odIndexList: number[],
  pointClusterMap: Map<number, number>,
  clusterPointMap: Map<number, number[]>,
  inAdjTable: Map<number, number[]>,  //  存储 <D点簇id，[O点簇id]>
  outAdjTable: Map<number, number[]>,  //  存储 <O点簇id，[D点簇id]>
  forceTreeLinks: ForceLink,
  forceTreeNodes: ForceNode,
}

export type ForceLink = Array<{source: number, target: number, isFake?: boolean}>;
export type ForceNode = Array<{name: string}>;