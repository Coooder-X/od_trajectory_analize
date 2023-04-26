import { computed, ComputedRef, onMounted, Ref, ref, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import { colorTable } from '@/color-pool';

export function useBrush({
  clusterLayerSvg, odPoints, project, unproject
}: {
  clusterLayerSvg: Ref<any | null>, odPoints: ComputedRef<number[][]>, project: Function, unproject: Function
}) {
  const store = useStore();
  const { getters } = store;
  const { inAdjTable, outAdjTable } = getters;
  const mapMode = getters.mapMode;
  let pointClusterMap = computed(() => getters.pointClusterMap);
  let clusterPointMap = computed(() => getters.clusterPointMap);
  let odIndexList = computed(() => getters.odIndexList);
  const x0 = ref(0);
  const y0 = ref(0);
  const x1 = ref(0);
  const y1 = ref(0);
  let selectedSvgs: Ref<any> = ref([]); //  存储被刷选的 circle svg
  let noSelectedSvgs: Ref<any> = ref([]); //  存储未被刷选的 circle svg
  let selectedODIdxs: Ref<number[]> = ref([]);  //  存储被刷选的 od 点索引
  let selectedClusterIdxs: Ref<number[]> = ref([]);  //  存储被刷选的 簇的索引

  console.log('clusterLayerSvg', clusterLayerSvg)
  let odCircles: any
  let brush: any
  let brushLayer: any

  // watch(clusterLayerSvg, (newValue, oldValue) => {
  //   if(newValue && !oldValue) {
  //     console.log('画布svg第一次初始化')
  //     brushLayer = clusterLayerSvg.value.append("g")
  //     brushLayer
  //       .attr("class", "brush")
  //       .call(brush)
  //       .style('display', 'none')
  //   }
  // })

  watch(clusterLayerSvg, () => {
    if(clusterLayerSvg.value) {
      // console.log([clusterLayerSvg.value.attr('width'), clusterLayerSvg.value.attr('height')])
      odCircles = clusterLayerSvg.value.selectAll("circle")
      brush = d3.brush()
        .extent([[0, 0], [800, 465]])  //  [clusterLayerSvg.value.attr('width'), clusterLayerSvg.value.attr('height')])
        .on("brush", brushed); // 设置刷取事件的回调函数
    
      // 在画布上添加刷取元素
      brushLayer = clusterLayerSvg.value.append("g")
      brushLayer
        .attr("class", "brush")
        .call(brush)
        .style('display', 'none')
    }
  }, {deep: false});

  // 定义刷取事件的回调函数
  function brushed(event: any) {
    selectedSvgs.value = []
    noSelectedSvgs.value = []
    selectedODIdxs.value = []
    selectedClusterIdxs.value = []
    //  记录【框框内的所有簇】包含的 od 点 id
    const selectedODSet: Set<number> = new Set();

    odCircles = clusterLayerSvg.value.selectAll("circle")
    // 获取刷取范围的坐标
    const selection = event.selection;
    console.log([unproject(selection[0]), unproject(selection[1])])
    x0.value = selection[0][0];
    y0.value = selection[0][1];
    x1.value = selection[1][0];
    y1.value = selection[1][1];

    /**  刷选实现：
     * （干2件事，1、得到选中的 od 和 簇 的数据，2、修改选中的 svg 颜色）
     * 1、先通过一次遍历 odCircles，得到在【选区内的所有簇 id】selectedClusterIdxs，以及这些簇包含的 od 点 id，
     * 2、再遍历 【选区内的所有簇 id】selectedClusterIdxs，得到【框框外与其有邻接关系的簇 id】，得到【框框内+关联到的框框外的所有簇】的 id
     * 3、根据所有已选簇 id，得到所有 od 点 id，再对 odCircles 遍历一次，统一修改颜色，并记录需要的数据。 
     */

    // 遍历所有的 circle 元素，i 是第几个 circle svg，要获取它的索引需要通过 odIndexList.value[i]
    odCircles.each(function (d: any, i: number) {
      // 获取当前 circle 的坐标
      const cx = project(d).x;
      const cy = project(d).y;

      // 判断当前 circle 是否在刷取范围内
      const inside = cx >= x0.value && cx <= x1.value && cy >= y0.value && cy <= y1.value;
      // 根据判断结果改变当前 circle 的颜色
      if (inside) {
        //  记录【框框内的所有簇】的簇 id
        selectedClusterIdxs.value.push(pointClusterMap.value.get(odIndexList.value[i]));  //  拿到在【框框内的所有簇 id】
      }
    });
    
    //  记录【框框内 + 框框外关联的 所有簇】包含的 簇 id
    const selectedClusterIdxSet: Set<number> = new Set(selectedClusterIdxs.value);
    selectedClusterIdxs.value = [...selectedClusterIdxSet];
    
    selectedClusterIdxs.value.forEach((clusterIdx: number) => {
      //  根据【框框内的所有簇 id】，得到框框外与其有邻接关系的簇 id
      let relatedClusterIds = new Set([...(inAdjTable.get(clusterIdx) || []), ...(outAdjTable.get(clusterIdx) || [])]);
      for(let relaCid of relatedClusterIds) {
        selectedClusterIdxSet.add(relaCid)
      }
    });
    //  更新 已选的所有 簇id 数组
    selectedClusterIdxs.value = [...selectedClusterIdxSet]
    //  根据已选的所有 簇id，得到已选的所有 od 点 id
    for(let cid of selectedClusterIdxSet) {
      const pInCluster = clusterPointMap.value.get(cid);
        for(let pid of pInCluster) {
          selectedODSet.add(pid);
        }
    }

    //  染色和收集 svg 数据
    odCircles.each(function(d: any, i: number) {
      if(selectedODSet.has(odIndexList.value[i])) { //  如果当前 circle svg 对应的 od 点在被选中的集合中
        selectedSvgs.value.push(d3.select(this));
        d3.select(this).style("fill", function(point: number[]) {
          const index = odIndexList.value[i];
          if(pointClusterMap.value.has(index))
            return colorTable[pointClusterMap.value.get(index)];
          return "lightblue";
        });
      } else {
        noSelectedSvgs.value.push(d3.select(this));
        d3.select(this).style("fill", "lightblue");
      }
    });
  }

  function setBrushLayerVisible(value: boolean) {
    console.log('setBrushLayerVisible', value)
    brushLayer
      .style('display', value ? '' : 'none');
  }

  return {
    setBrushLayerVisible,
    selectedSvgs,
    noSelectedSvgs,
    selectedODIdxs,
    selectedClusterIdxs,
  }
}

export function debounce(callback: Function, delay: number) {
  var timer: number | undefined = undefined
  return function() { //  下面用了arguments，所以这里要返回function而不是箭头函数
    var firstClick = !timer
    if(!firstClick) {
      clearTimeout(timer)
    }
    timer = setTimeout(() => {
      timer = undefined
      if(!firstClick)
        callback(...arguments)
    }, delay)
  }
}

//  获得 gis 中所有 OD 点的 svg d3 选择集的 hooks
export function useGetOdCircles() {
  const store = useStore();
  const { getters } = store;
  const odCircles: Ref<any> = ref(null);
  const clusterLayerSvg: ComputedRef<any | null> = computed(
    () => getters.clusterLayerSvg
  );

  watch(clusterLayerSvg, () => {
    if(clusterLayerSvg.value) {
      odCircles.value = clusterLayerSvg.value.selectAll("circle")
    }
  }, {deep: false, immediate: true});

  return {
    odCircles,
  };
}

//  
export function useGetCircleByCluster() {
  const store = useStore();
  const { getters } = store;
  const { odCircles } = useGetOdCircles();
  const odIndexList = computed(() => getters.odIndexList);
  const pointClusterMap = computed(() => getters.pointClusterMap);

  //  输入簇 id，得到一个 d3 选择集，包含在这个簇中的 OD 点 svg
  function getCircleByClusterId(clusterId: number) {
    const result = odCircles.value.filter(function(d: any, i: number) {
      const index = odIndexList.value[i];
      if (pointClusterMap.value.get(index) === clusterId) {
        return true;
      }
    });
    console.log(result)
    return result;
  }

  return {
    getCircleByClusterId
  };
}