import { GlobalState } from '@/global-interface';
import { MapViewState } from '@/map-interface';
import axios from 'axios';
import { ActionContext } from 'vuex';

const initState: GlobalState = {
  pointsExist: false,
  timeScope: [1, 2],
  dateScope: [8, 10],
  odPoints: [],
  odIndexList: [],
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
    }
  },
  modules: {},
};

export default globalModule;