<template>
  <div class="map-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <div class="map-view-content">
      <div class="side-bar">
        <div class="side-bar-button" @click="toggleClusterLayer($event)">
          <el-tooltip
            :content="clusterLayerShow? '隐藏OD点' : '显示OD点'"
            placement="right">
            <img src="@/assets/location_fill.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button" @click="()=>{}">
          <el-tooltip
            content="聚类选项"
            placement="right">
            <img src="@/assets/chart-bubble.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="刷选"
            placement="right">
            <img src="@/assets/框选.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="点选"
            placement="right">
            <img src="@/assets/pointer.svg" alt="" />
          </el-tooltip>
        </div><div class="side-bar-button">
          <el-tooltip
            content="显示OD流"
            placement="right">
            <img src="@/assets/pin-distance-line-active.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="显示轨迹"
            placement="right">
            <img src="@/assets/路径分析.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="过滤"
            placement="right">
            <img src="@/assets/filter.svg" alt="" />
          </el-tooltip>
        </div>
      </div>
      <map-comp class="map-comp"></map-comp>
    </div>
  </div>
</template>

<script>
/* eslint-disable */
import { defineComponent, computed, onMounted } from "vue";
import { useStore } from 'vuex';
import ViewHeader from "../ViewHeader";
import MapComp from "../MapComp";
import { MapMode, MapModeTooltip } from '@/map-interface.ts'
import {
  Document,
  Menu as IconMenu,
  Location,
  Setting,
} from "@element-plus/icons-vue";

export default defineComponent({
  components: {
    MapComp,
    ViewHeader,
    Document,
    IconMenu,
    Location,
    Setting,
  },
  name: "MapView",
  props: {},
  setup() {
    const store = useStore();
    const clusterLayerShow = computed(() => store.state.layers.clusterLayerShow);

    onMounted(() => {
      store.dispatch('helloWorld');
    })

    const chooseMode = (index) => {
      console.log("chooseMode", index);
    };

    // 切换是否显示轨迹点图层
    const toggleClusterLayer = () => {
      store.commit('setClusterLayerShow', !store.state.layers.clusterLayerShow);
    }

    return {
      clusterLayerShow,
      toggleClusterLayer,
      chooseMode,
      MapMode,
      MapModeTooltip
    };
  },
});
</script>

<style scoped>
img {
  height: 30px;
  width: 30px;
}
img:hover {
  fill: aqua;
}
.map-view {
  --menu-width: 50px;
  height: 500px;
  width: 850px;
  background-color: white;
}

.map-view-content {
  height: calc(100% - var(--header-height));
  display: flex;
}

.side-bar {
  padding-top: 5px;
  height: calc(100% - var(--header-height));
  width: var(--menu-width);
  position: absolute;
}

.side-bar-button {
  display: flex;
  height: 50px;
  align-items: center;
  justify-content: center;
}
.side-bar-button:hover {
  cursor: pointer;
  background-color: aliceblue;
}

.map-comp {
  margin-left: var(--menu-width);
  height: 100%;
  width: calc(100% - var(--menu-width));
}
</style>
