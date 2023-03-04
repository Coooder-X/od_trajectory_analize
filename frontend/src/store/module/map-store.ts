import { ActionContext } from 'vuex';

// const initState: DataSetState = {
// 	dataTree: null,
// 	loading: false,
// 	file: {},
// };
const initState = {
    
}

const mapModule = {
  state: {
    ...initState
  },
  mutations: {
    // getDataSet(state: DataSetState, payload: any) {
    //   state.dataTree = payload;
    // },
  },
  actions: {
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
  modules: {},
};

export default mapModule;