import { createStore } from 'vuex';
// import axios from 'axios';
import mapModule from './module/map-store';

const initState = {
  
};

const store = createStore({
  state: {
    ...initState
  },
  mutations: {
    // getPath(state: RootState, payload: any) {
    //   state.path = payload
    // },
  },
  actions: {
    // setPath(context, value) {
    //   context.commit('getPath', value)
    //   console.log('path', value);
    // },
  },
  modules: {
    layers: mapModule
  },
});

export default store;