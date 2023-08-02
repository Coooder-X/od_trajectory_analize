<template>
  <div class="grid-line" :id="`speed_${TrjId}`" :key="`speed_${TrjId}`">
    <div class="line-head">{{ `${TrjId}: ` }}</div>
    <div v-for="value in localSpeedList" :key="value" class="grid" :style="{'backgroundColor': colorMap[value]}"></div>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, nextTick, onMounted, PropType, Ref, ref, watch } from "vue";
import * as d3 from 'd3';

export default defineComponent({
  components: {
  },
  name: "HeatGrid",
  props: {
    TrjId: Number,
    speedList: Array as PropType<Array<number>>
  },
  setup(props) {
    // const speedData = props.speedData;
    const { TrjId } = props;
    console.log('props', props)
    const localSpeedList = computed(() => props.speedList);
    // const TrjId = computed(() => props.TrjId);
    const side = 20;
    const colorMap = {
      // 1: 'rgb(255, 222, 173)',
      // 2: 'rgb(182, 219, 182)',
      // 3: 'rgb(146, 201, 146)',
      // 4: 'rgb(73, 164, 73)',
      1: 'rgb(227, 216, 164)',
      2: 'rgb(171, 204, 147)',
      3: 'rgb(116, 191, 130)',
      4: 'rgb(60, 179, 113)',
    }

    watch(localSpeedList, () => {
    // onMounted(() => {
      // 设置颜色比例尺
      nextTick(() => {
        var color = d3.scaleLinear()
        .domain([0, 4])
        .range(["white", "orange"]);
  
        console.log('dom', d3.select(`#speed_${TrjId}`))
    
        const svg = d3.select(`#speed_${TrjId}`)
          .selectAll('svg')//.exit().remove()
          .append("svg")
          .attr("width", 680)
          .attr("height", 50)//.append('g').attr("width", 680)
          // .attr("height", 50);

        console.log('svg2', svg)

        console.log('watch', localSpeedList.value)
        // 绑定数据并创建正方形
        var squares = svg.selectAll("rect")
          .data([1,2,3,3,3,3])
          .enter()
          .append("rect")

        console.log('squares', squares)

        squares
          .attr("x", function(d: any, i: number) {
            console.log(i)
              return i * side; // 根据索引计算x坐标
          })
          // .attr('stroke', 'black')
          // .attr('stroke-width', '0.5px')
          .attr("y", 0) // y坐标固定为0
          .attr("width", side) // 宽度等于边长
          .attr("height", side) // 高度等于边长
          .attr("fill", function(d: any) {
            console.log(color(d))
            return color(d); // 根据数据值计算颜色
          });
      });
    // });
    }, {deep: true, immediate: true});
    
    return {
      TrjId,
      localSpeedList,
      colorMap,
    }
  },
});
</script>

<style scoped>
.grid-line {
  width: 100%;
  height: 50px;
  margin-top: 6px;
  /* background-color: rgb(228, 255, 205); */
  display: flex;
  flex-wrap: nowrap;
  justify-content: flex-start;
  align-items: center;
}

.grid {
  width: 20px;
  height: 20px;
  border: solid 0.5px gray;
  flex-shrink: 0;
}

.line-head {
  width: 50px;
  text-align: end;
  /* margin-left: 5px; */
  margin-right: 5px;
}

/* .grid-line > .grid-list {

} */
</style>