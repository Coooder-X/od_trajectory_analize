import { computed, ComputedRef, onMounted, Ref, ref, watch } from 'vue';
import { MapMode } from '@/map-interface';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import { colorTable } from '@/color-pool';

export function useBrush({
  clusterLayerSvg, project, unproject
}: {
  clusterLayerSvg: Ref<any | null>, project: Function, unproject: Function
}) {
  const store = useStore();
  const { getters } = store;
  const { inAdjTable, outAdjTable } = getters;
  const mapMode = getters.mapMode;
  const pointClusterMap = computed(() => getters.pointClusterMap);
  const clusterPointMap = computed(() => getters.clusterPointMap);
  const odIndexList = computed(() => getters.odIndexList);
  const x0 = ref(0);
  const y0 = ref(0);
  const x1 = ref(0);
  const y1 = ref(0);
  const selectedSvgs: Ref<any> = ref([]); //  存储被刷选的 circle svg
  const noSelectedSvgs: Ref<any> = ref([]); //  存储未被刷选的 circle svg
  const selectedODIdxs: Ref<number[]> = ref([]);  //  存储被刷选的 od 点索引
  const selectedClusterIdxs: Ref<number[]> = ref([]);  //  存储被刷选的 簇的索引
  const selectedClusterIdxsInBrush: Ref<number[]> = ref([]);

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
      console.log('点个数 =', odCircles.size())
      brush = d3.brush()
        .extent([[0, 0], [800, 465]])  //  [clusterLayerSvg.value.attr('width'), clusterLayerSvg.value.attr('height')])
        .on("brush", brushed) // 设置刷取事件的回调函数
        .on('end', endBrush)
    
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
    selectedClusterIdxsInBrush.value = [...selectedClusterIdxSet];
    
    selectedClusterIdxs.value.forEach((clusterIdx: number) => {
      //  根据【框框内的所有簇 id】，得到框框外与其有邻接关系的簇 id
      const relatedClusterIds = new Set([...(inAdjTable.get(clusterIdx) || []), ...(outAdjTable.get(clusterIdx) || [])]);
      for(const relaCid of relatedClusterIds) {
        selectedClusterIdxSet.add(relaCid)
      }
    });
    //  更新 已选的所有 簇id 数组
    selectedClusterIdxs.value = [...selectedClusterIdxSet]
    //  根据已选的所有 簇id，得到已选的所有 od 点 id
    for(const cid of selectedClusterIdxSet) {
      const pInCluster = clusterPointMap.value.get(cid);
        for(const pid of pInCluster) {
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
    odCircles.attr('stroke', 'black');
  }

  function endBrush() {
    //  确定选区停止拖拽后，未选中的 OD 点，不展示，防止画面太乱干扰分析
    noSelectedSvgs.value.forEach((item: any, i: number) => {
      if (i % 1000 === 1)
        console.log('item', item)
      item
        .style('fill', 'transparent')
        .attr('stroke', 'transparent')
    })
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
    selectedClusterIdxsInBrush,
  }
}

export function debounce(callback: Function, delay: number) {
  let timer: number | undefined = undefined
  return function() { //  下面用了arguments，所以这里要返回function而不是箭头函数
    const firstClick = !timer
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
// export function useGetOdCircles() {
//   const store = useStore();
//   const { getters } = store;
//   const odCircles: Ref<any> = ref(null);
//   const clusterLayerSvg: ComputedRef<any | null> = computed(
//     () => getters.clusterLayerSvg
//   );

//   watch(clusterLayerSvg, () => {
//     if(clusterLayerSvg.value) {
//       odCircles.value = clusterLayerSvg.value.selectAll("circle")
//     }
//   }, {deep: false, immediate: true});

//   return {
//     odCircles,
//   };
// }

//  
export function useGetCircleByCluster() {
  const store = useStore();
  const { getters } = store;
  const odIndexList = computed(() => getters.odIndexList);
  const pointClusterMap = computed(() => getters.pointClusterMap);

  //  输入簇 id，得到一个 d3 选择集，包含在这个簇中的 OD 点 svg
  function getCircleByClusterId(odCircles: any, clusterId: number) {
    // let result: Ref<any> = ref(null);
    // result.value = 
    return odCircles.filter(function(d: any, i: number) {
      const index = odIndexList.value[i];
      if (pointClusterMap.value.get(index) === clusterId) {
        return true;
      }
    });
    // console.log(result)
    // return result;
  }

  return {
    getCircleByClusterId
  };
}

export function useDrawODPath(project: Function, clusterLayerSvg: Ref<any>) {
  
  const store = useStore();
  const { getters } = store;
  const { filteredOutAdjTable } = getters;
  type Coord = {x: number, y: number};
  const odPairList: Ref<Array<[Coord, Coord]>> = ref([]);
  let links: any = null;
  let line: any = null;
  const svgMarker: Ref<any> = ref(null);
  const odArrows: Ref<any> = ref(null);
  const mapMode = computed(() => getters.mapMode);

  function setMarker(svgPath: any) {
    const marker = svgPath //  设置箭头
      .append("svg:defs")
      .append("marker")
      .attr("id", "od_arrow") //设置箭头的 id，用于引用
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 3) //设置箭头距离节点的距离
      .attr("refY", 0) //设置箭头在 y 轴上的偏移量
      .attr("orient", "auto") //设置箭头随边的方向旋转
      .attr("markerWidth", 2.8) //设置箭头的宽度
      .attr("markerHeight", 2.8) //设置箭头的高度
      .attr("xoverflow", "visible");
    svgMarker.value = marker;
  }

  function drawODPath(cidCenterMap: Map<number, [number, number]>) {
    if (!mapMode.value.has(MapMode.CHOOSE_POINT)) {
      d3.select('#paths').selectAll('path').remove();
      d3.selectAll('#one_od').remove();
      return;
    }

    const odPairs = JSON.parse(sessionStorage.getItem("odPairs")!);
    odPairList.value = [];
    for (const odPair of odPairs) {
      const [srcCid, tgtCid] = odPair;
      const oCenterCoord = cidCenterMap.get(srcCid)!;
      const dCenterCoord = cidCenterMap.get(tgtCid)!;
      odPairList.value.push([
        {x: oCenterCoord[0], y: oCenterCoord[1]},
        {x: dCenterCoord[0], y: dCenterCoord[1]}
      ]);
    }

    // for(let item of filteredOutAdjTable) {
    //   let [key, value]: [number, number[]] = item;
    //   if(Object.keys(value).length === 0)
    //     continue;
    //   const oCenterCoord = cidCenterMap.get(key)!;
    //   value.forEach((cid: number) => {
    //     const dCenterCoord = cidCenterMap.get(cid)!;
    //     odPairList.value.push([
    //       {x: oCenterCoord[0], y: oCenterCoord[1]},
    //       {x: dCenterCoord[0], y: dCenterCoord[1]}
    //     ]);
    //   })
    // };

    const g = d3.select('#paths')

    // g.select('path').remove();
    console.log('path', g.select('path'))

    line = d3.line()
      .x(function(d: any) { return project([d.x, d.y]).x; })
      .y(function(d: any) { return project([d.x, d.y]).y; })
      .curve(d3.curveBasis);
    
    // 使用选择集绑定数据和path元素
    links = g.selectAll("path")
      .data(odPairList.value);   //  绑定odPairList数组

    links.enter()
      .append("path")
      .attr("d", line)
      .attr('id', 'one_od')
      .attr("marker-end", "url(#od_arrow)")
      .attr("stroke", 'rgb(255 108 55)')
      .attr('opacity', 1)
      .attr("stroke-width", 4)
      .attr("stroke-dasharray", "1,0") // 设置虚线间隔为1px和0px

    // 使用exit()函数删除多余的path元素，如果有的话
    links.exit().remove();    
    odArrows.value = clusterLayerSvg.value.select('#paths').selectAll('path');
  }

  function moveOdPath() {
    if (!mapMode.value.has(MapMode.CHOOSE_POINT)) {
      d3.select('#paths').selectAll('path').remove();
      d3.selectAll('#one_od').remove();
      return;
    }
    odArrows.value.attr('d', line);
  }

  function updateArrow() {
    if (!mapMode.value.has(MapMode.CHOOSE_POINT)) {
      d3.select('#paths').selectAll('path').remove();
      d3.selectAll('#one_od').remove();
      return;
    }
    svgMarker.value
      .append("path")
      .attr("d", "M 0,-5 L 10 ,0 L 0,5") //使用绝对坐标来绘制三角形
      .attr('fill', 'rgb(255 108 55)')
      .attr('stroke', 'none')
      .attr("stroke-dasharray", "1,0") // 设置虚线间隔为1px和0px
  }

  return {
    setMarker,
    drawODPath,
    moveOdPath,
    odArrows,
    odPairList,
    updateArrow,
  }
}