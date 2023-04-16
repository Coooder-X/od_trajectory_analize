import { MapMode, MapViewState } from '@/map-interface';
import axios from 'axios';
import { ActionContext } from 'vuex';

const initState: MapViewState = {
  clusterLayerSvg: null,
  odLayerSvg: null,
  trjLayerSvg: null,
  clusterLayerShow: false,
  codLayerShow: false,
  trjLayeShow: false,
  mapMode: new Set([MapMode.INIT]),
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
    toggleMapMode(state: MapViewState, payload: string) {
      if(state.mapMode.has(payload)) {
        state.mapMode.delete(payload)
      } else
        state.mapMode.add(payload);
    }
  },
  actions: {
    helloWorld(context: ActionContext<{}, {}>) {
      axios.get('/api').then((res) => {
				console.log(res);
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
    mapMode: (state: MapViewState) => {
      return state.mapMode;
    }
  },
  modules: {},
};

export default mapModule;