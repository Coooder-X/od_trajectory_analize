import { ForceLink, ForceNode, GlobalState } from '@/global-interface';
import axios from 'axios';
import { ActionContext } from 'vuex';

const initState: GlobalState = {
  pointsExist: false,
  dateScope: [1, 2],
  timeScope: [8, 10],
  odPoints: [], //  地图上存在的所有 od 点，因此并不是全量的数据。//在后端中，目前全量数据是从一天的 od 点中取一部分点
  odIndexList: [],
  pointClusterMap: new Map(),
  clusterPointMap: new Map(),
  inAdjTable: new Map(),
  outAdjTable: new Map(),
  forceTreeLinks: [],
  forceTreeNodes: [],
  selectedODIdxs: [],
  selectedClusterIdxs: [],
  cidCenterMap: new Map(),
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
    setAllODPoints(state: GlobalState, payload: Array<[]>) {
      state.odPoints = payload;
      console.log('set points', state.odPoints)
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
  },
  actions: {
    getAllODPoints(context: ActionContext<{}, {}>) {
      context.commit('setPointsExist', false);
      axios.get('/api/getTotalODPoints').then((res) => {
        console.log('getAllODPoints', res, res.status === 200);
        context.commit('setAllODPoints', res.data);
        context.commit('setPointsExist', res.status === 200);
      })
    },
    getODPointsFilterByHour(context: ActionContext<{}, {}>, params: any) {
      axios.get('/api/getODPointsFilterByHour', params).then((res) => {
        console.log('getODPointsFilterByHour', res, res.status === 200);
        //  设置 od 点的坐标数组和 index 序号数组
        context.commit('setAllODPoints', res.data['od_points']);
        context.commit('setODIndexList', res.data['index_lst']);
        context.commit('setPointsExist', res.status === 200);
      })
    },
    getClusteringResult(context: ActionContext<{}, {}>, params: any) {
      axios.get('/api/getClusteringResult', params).then((res) => {
        console.log('getClusteringResult', res, res.status === 200);
        //  设置 od 点的坐标数组和 index 序号数组
        context.commit('setPointClusterMap', res.data['point_cluster_dict']);
        context.commit('setClusterPointMap', res.data['cluster_point_dict']);
        context.commit('setAllODPoints', res.data['od_points']);
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
      });
    },
    getCidCenterMap(context: ActionContext<{}, {}>, params: any) {
      console.log('getCidCenterMap')
      axios({
        method: 'post',
        url: '/api/getClusterCenter',
        data: params,
      }).then((res) => {
        console.log('getCidCenterMap gettget')
        context.commit('setCidCenterMap', res.data['cid_center_coord_dict']);
      });
    },
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
    timeScope: (state: GlobalState) => {
      return state.timeScope;
    },
    pointClusterMap: (state: GlobalState) => {
      return state.pointClusterMap;
    },
    clusterPointMap: (state: GlobalState) => {
      return state.clusterPointMap;
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
  },
  modules: {},
};

export default globalModule;