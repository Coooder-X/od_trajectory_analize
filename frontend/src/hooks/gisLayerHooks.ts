import { computed, ComputedRef, Ref, ref, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import { colorTable } from '@/color-pool';

export function useBrush({
  clusterLayerSvg, odPoints, project
}: {
  clusterLayerSvg: Ref<any | null>, odPoints: ComputedRef<number[][]>, project: Function
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
    odCircles = clusterLayerSvg.value.selectAll("circle")
    // 获取刷取范围的坐标
    const selection = event.selection;
    console.log(selection)
    x0.value = selection[0][0];
    y0.value = selection[0][1];
    x1.value = selection[1][0];
    y1.value = selection[1][1];

    /**  刷选实现：
     * （干2件事，1、得到选中的 od 和 簇 的数据，2、修改选中的 svg 颜色）
     * 1、先通过一次遍历 odCircles，得到在【选区内的所有簇 id】selectedClusterIdxs，以及这些簇包含的 od 点 id，
     * 2、再遍历 【选区内的所有簇 id】selectedClusterIdxs，得到【框框外与其有邻接关系的簇 id】，得到【关联到的框框外的所有簇】包含的 od 点 id
     * 3、得到所有 od 点 id，再对 odCircles 遍历一次，统一修改颜色，并记录需要的数据。 
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
        selectedClusterIdxs.value.push(pointClusterMap.value.get(odIndexList.value[i]));  //  拿到在【框框内的所有簇 id】
      }
    });
    
    const selectedODSet: Set<number> = new Set();
    //  记录【框框内的所有簇】包含的 od 点 id
    for(let cid of selectedClusterIdxs.value) {
      const pInCluster = clusterPointMap.value.get(cid);
      for(let pid of pInCluster) {
        selectedODSet.add(pid);
      }
    }
    
    selectedClusterIdxs.value.forEach((clusterIdx: number) => {
      //  根据【框框内的所有簇 id】，得到框框外与其有邻接关系的簇 id
      let relatedClusterIds = new Set([...(inAdjTable.get(clusterIdx) || []), ...(outAdjTable.get(clusterIdx) || [])]);
        //  记录【关联到的框框外的所有簇】包含的 od 点 id
        for(let cid of relatedClusterIds) {
          const pInCluster = clusterPointMap.value.get(cid);
          for(let pid of pInCluster) {
            selectedODSet.add(pid);
          }
        }
    });
    //  od 点 id 数组 selectedODIdxs 去重
    selectedODIdxs.value = [...selectedODSet]
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
  }
}
