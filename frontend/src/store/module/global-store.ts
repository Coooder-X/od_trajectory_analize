import { ForceLink, ForceNode, GlobalState } from '@/global-interface';
import axios from 'axios';
import { ActionContext } from 'vuex';

const initState: GlobalState = {
  pointsExist: false,
  timeScope: [1, 2],
  dateScope: [8, 10],
  odPoints: [],
  odIndexList: [],
  pointClusterMap: new Map(),
  clusterPointMap: new Map(),
  inAdjTable: new Map(),
  outAdjTable: new Map(),
  forceTreeLinks: [],
  forceTreeNodes: [],
  selectedODIdxs: [],
  selectedClusterIdxs: [],
}

const globalModule = {
  namespace: true,
  state: {
    ...initState
  },
  mutations: {
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
    // createCategory(context: ActionContext<{}, {}>, params: any) {
    //   axios.post('/api/dataset/createCategory', params);
    // },
    // getFile(context: ActionContext<{}, {}>, params: any) {
    //   console.log('getFile params', params);
    //   axios.get('/api/dataset/getFile', params).then((res) => {
    //     console.log('getFile data', res.data);
    //     context.commit('getFile', res.data);
    //   });
    // }
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
  },
  modules: {},
};

export default globalModule;