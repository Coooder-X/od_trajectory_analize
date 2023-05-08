import { ForceLink, ForceNode, GlobalState } from '@/global-interface';
import axios from 'axios';
import { ActionContext } from 'vuex';

const initState: GlobalState = {
  pointsExist: false,
  month: 5,
  dateScope: [0, 1],
  timeScope: [8, 10],
  odPoints: [], //  当天的全量 OD 点数据。//在后端中，目前全量数据是从一天的 od 点中取一部分点
  partOdPoints: [], //  地图上存在的所有 od 点，因此并不是全量的数据，是小时段筛选后的。
  odIndexList: [],
  pointClusterMap: new Map(),
  clusterPointMap: new Map(),
  partClusterPointMap: new Map(),
  inAdjTable: new Map(),
  outAdjTable: new Map(),
  filteredOutAdjTable: new Map(), //  刷选过滤后的 出边邻接表
  forceTreeLinks: [],
  forceTreeNodes: [],
  selectedODIdxs: [],
  selectedClusterIdxs: [],
  cidCenterMap: new Map(),
  communityGroup: new Map(),
  withSpaceDist: false,
  colorTable: [],
}

const globalModule = {
  namespace: true,
  state: {
    ...initState
  },
  mutations: {
    setTimeScope(state: GlobalState, payload: [number, number]) {
      state.timeScope = payload;
    },
    setDateScope(state: GlobalState, payload: [number, number]) {
      state.dateScope = payload;
    },
    setAllODPoints(state: GlobalState, payload: Array<[]>) {
      // sessionStorage.setItem('odPoints', JSON.stringify(
      //   payload
      // ));
      state.odPoints = payload;
      // console.log('set points', state.odPoints)
    },
    setPartODPoints(state: GlobalState, payload: Array<number[]>) {
      state.partOdPoints = payload;
    },
    setPointsExist(state: GlobalState, payload: Boolean) {
      state.pointsExist = payload;
    },
    setODIndexList(state: GlobalState, payload: number[]) {
      state.odIndexList = payload;
    },
    setPointClusterMap(state: GlobalState, payload: {[key: number]: number}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key)
        state.pointClusterMap.set(k, payload[k]);
      })
    },
    setClusterPointMap(state: GlobalState, payload: {[key: number]: number[]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key)
        state.clusterPointMap.set(k, payload[k]);
      })
    },
    setPartClusterPointMap(state: GlobalState, payload: {[key: number]: number[]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key)
        state.partClusterPointMap.set(k, payload[k]);
      })
    },
    setInAdjTable(state: GlobalState, payload: {[key: string]: number[]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key);
        state.inAdjTable.set(k, payload[key]);
      });
    },
    setOutAdjTable(state: GlobalState, payload: {[key: string]: number[]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key);
        state.outAdjTable.set(k, payload[key]);
      });
    },
    setFilteredOutAdjTable(state: GlobalState, payload: {[key: string]: number[]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key);
        state.filteredOutAdjTable.set(k, payload[key]);
      });
    },
    setForceTreeLinks(state: GlobalState, payload: ForceLink) {
      state.forceTreeLinks = payload;
    },
    setForceTreeNodes(state: GlobalState, payload: ForceNode) {
      state.forceTreeNodes = payload;
    },
    setSelectedODIdxs(state: GlobalState, payload: number[]) {
      state.selectedODIdxs = payload;
    },
    setSelectedClusterIdxs(state: GlobalState, payload: number[]) {
      state.selectedClusterIdxs = payload;
    },
    setCidCenterMap(state: GlobalState, payload: {[key: string]: [number, number]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key);
        state.cidCenterMap.set(k, payload[key]);
      });
    },
    setCommunityGroup(state: GlobalState, payload: {[key: number]: string[]}) {
      Object.keys(payload).forEach((key: string) => {
        let k = parseInt(key);
        if (payload[k].length > 0) {
          state.communityGroup.set(k, payload[k]);
        }
      });
    },
    setWithSpaceDist(state: GlobalState, payload: Boolean) {
      state.withSpaceDist = payload;
    },
    setColorTable(state: GlobalState, payload: string[]) {
      state.colorTable = payload;
    },
    setMonth(state: GlobalState, payload: number) {
      state.month = payload;
    },
  },
  actions: {
    getAllODPoints(context: ActionContext<{}, {}>, params: any) {
      context.commit('setPointsExist', false);
      // axios.get('/api/getTotalODPoints').then((res) => {
      //   console.log('getAllODPoints', res, res.status === 200);
      //   context.commit('setAllODPoints', res.data);
      //   context.commit('setPointsExist', res.status === 200);
      // })
      axios.get('/api/getODPointsFilterByDayAndHour', params).then((res) => {
        //  设置 od 点的坐标数组和 index 序号数组
        const month = Object.keys(res.data)[0];
        console.log('month', month)
        context.commit('setAllODPoints', res.data[month]['od_points']);
        context.commit('setODIndexList', res.data[month]['index_lst']);
        context.commit('setPointsExist', res.status === 200);
      })
    },
    getODPointsFilterByHour(context: ActionContext<{}, {}>, params: any) {
      axios.get('/api/getODPointsFilterByHour', params).then((res) => {
        console.log('getODPointsFilterByHour', res, res.status === 200);
        //  设置 od 点的坐标数组和 index 序号数组
        context.commit('setPartODPoints', res.data['part_od_points']);
        context.commit('setODIndexList', res.data['index_lst']);
        context.commit('setPointsExist', res.status === 200);
      })
    },
    getODPointsFilterByDayAndHour(context: ActionContext<{}, {}>, params: any) {
      axios.get('/api/getODPointsFilterByDayAndHour', params).then((res) => {
        //  设置 od 点的坐标数组和 index 序号数组
        const month = Object.keys(res.data)[0];
        console.log('month', month)
        context.commit('setPartODPoints', res.data[month]['od_points']);
        context.commit('setODIndexList', res.data[month]['index_lst']);
      })
    },
    getClusteringResult(context: ActionContext<{}, {}>, params: any) {
      axios.get('/api/getClusteringResult', params).then((res) => {
        console.log('getClusteringResult', res, res.status === 200);
        //  设置 od 点的坐标数组和 index 序号数组
        context.commit('setPointClusterMap', res.data['point_cluster_dict']);
        context.commit('setClusterPointMap', res.data['cluster_point_dict']);
        context.commit('setPartClusterPointMap', res.data['part_cluster_point_dict']);
        context.commit('setPartODPoints', res.data['part_od_points']);
        context.commit('setODIndexList', res.data['index_lst']);
        // context.commit('setForceTreeLinks', res.data['json_adj_table']);
        // context.commit('setForceTreeNodes', res.data['json_nodes']);
        context.commit('setInAdjTable', res.data['in_adj_table']);
        context.commit('setOutAdjTable', res.data['out_adj_table']);
      })
    },
    getLineGraph(context: ActionContext<{}, {}>, params: any) {
      axios({
        method: 'post',
        url: '/api/getLineGraph',
        data: params,
      }).then((res) => {
        console.log(res)
        context.commit('setForceTreeLinks', res.data['force_edges']);
        context.commit('setForceTreeNodes', res.data['force_nodes']);
        context.commit('setFilteredOutAdjTable', res.data['filtered_adj_dict']);
        context.commit('setCidCenterMap', res.data['cid_center_coord_dict']);
        context.commit('setCommunityGroup', res.data['community_group']);
      });
    },
    // getCidCenterMap(context: ActionContext<{}, {}>, params: any) {
    //   console.log('getCidCenterMap')
    //   axios({
    //     method: 'post',
    //     url: '/api/getClusterCenter',
    //     data: params,
    //   }).then((res) => {
    //     console.log('getCidCenterMap gettget')
    //     context.commit('setCidCenterMap', res.data['cid_center_coord_dict']);
    //   });
    // },
    // createCategory(context: ActionContext<{}, {}>, params: any) {
    //   axios.post('/api/dataset/createCategory', params);
    // },
  },
  getters: {
    pointsExist: (state: GlobalState) => {
      console.log('getters', state.pointsExist)
      return state.pointsExist;
    },
    odPoints: (state: GlobalState) => {
      return state.odPoints;
    },
    partOdPoints: (state: GlobalState) => {
      return state.partOdPoints;
    },
    timeScope: (state: GlobalState) => {
      return state.timeScope;
    },
    dateScope: (state: GlobalState) => {
      return state.dateScope;
    },
    pointClusterMap: (state: GlobalState) => {
      return state.pointClusterMap;
    },
    clusterPointMap: (state: GlobalState) => {
      return state.clusterPointMap;
    },
    partClusterPointMap: (state: GlobalState) => {
      return state.partClusterPointMap;
    },
    odIndexList: (state: GlobalState) => {
      return state.odIndexList;
    },
    inAdjTable: (state: GlobalState) => {
      return state.inAdjTable;
    },
    outAdjTable: (state: GlobalState) => {
      return state.outAdjTable;
    },
    filteredOutAdjTable: (state: GlobalState) => {
      return state.filteredOutAdjTable;
    },
    forceTreeLinks: (state: GlobalState) => {
      return state.forceTreeLinks;
    },
    forceTreeNodes: (state: GlobalState) => {
      return state.forceTreeNodes;
    },
    selectedODIdxs: (state: GlobalState) => {
      return state.selectedODIdxs;
    },
    selectedClusterIdxs: (state: GlobalState) => {
      return state.selectedClusterIdxs;
    },
    cidCenterMap: (state: GlobalState) => {
      return state.cidCenterMap;
    },
    communityGroup: (state: GlobalState) => {
      return state.communityGroup;
    },
    withSpaceDist: (state: GlobalState) => {
      return state.withSpaceDist;
    },
    colorTable: (state: GlobalState) => {
      return state.colorTable;
    },
    month: (state: GlobalState) => {
      return state.month;
    },
  },
  modules: {},
};

export default globalModule;