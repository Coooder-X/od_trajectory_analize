export interface GlobalState {
  pointsExist: Boolean,
  timeScope: [number, number],
  dateScope: [number, number],
  odPoints: Array<[]>,
  odIndexList: number[],
  pointClusterMap: Map<number, number>,
  clusterPointMap: Map<number, number[]>,
  inAdjTable: Map<number, number[]>,  //  存储 <D点簇id，[O点簇id]>
  outAdjTable: Map<number, number[]>,  //  存储 <O点簇id，[D点簇id]> (全量)
  filteredOutAdjTable: Map<number, number[]>, //  筛选过
  forceTreeLinks: ForceLink,
  forceTreeNodes: ForceNode,
  selectedODIdxs: number[],
  selectedClusterIdxs: number[],
  cidCenterMap: Map<number, [number, number]>,  // <簇id, [lon, lat]>，簇中心点坐标
}

export type ForceLink = Array<{source: number, target: number, isFake?: boolean, value: number}>;
export type ForceNode = Array<{name: string, trjNum?: number, avgLen?: number}>;