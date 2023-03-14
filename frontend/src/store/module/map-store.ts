import { MapViewState } from '@/map-interface';
import axios from 'axios';
import { ActionContext } from 'vuex';

const initState: MapViewState = {
  pointsExist: false,
  clusterLayerSvg: null,
  odLayerSvg: null,
  trjLayerSvg: null,
  clusterLayerShow: false,
  codLayerShow: false,
  trjLayeShow: false,
  data: {
    totalODPoints: [],
  }
}

const mapModule = {
  namespace: true,
  state: {
    ...initState
  },
  mutations: {
    setClusterLayerSvg(state: MapViewState, payload: SVGAElement) {
      state.clusterLayerSvg = payload;
    },
    setClusterLayerShow(state: MapViewState, payload: Boolean) {
      state.clusterLayerShow = payload
      state.clusterLayerSvg?.attr('visibility', !payload? 'hidden' : 'visible')
    },
    setAllODPoints(state: MapViewState, payload: Array<[]>) {
      state.data.totalODPoints = payload;
      console.log('set points', state.data.totalODPoints)
    },
    setPointsExist(state: MapViewState, payload: Boolean) {
      state.pointsExist = payload;
    }
  },
  actions: {
    helloWorld(context: ActionContext<{}, {}>) {
      axios.get('/api').then((res) => {
				console.log(res);
			})
    },
    getAllODPoints(context: ActionContext<{}, {}>) {
      context.commit('setPointsExist', false);
      console.log(initState.pointsExist)
      axios.get('/api/getTotalODPoints').then((res) => {
				console.log('getAllODPoints', res, res.status === 200);
        context.commit('setAllODPoints', res.data);
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
    pointsExist: (state: MapViewState) => {
      console.log('getters', state.pointsExist)
      return state.pointsExist;
    },
    totalODPoints: (state: MapViewState) => {
      return state.data.totalODPoints;
    }
  },
  modules: {},
};

export default mapModule;