<template>
  <div class="map-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <div class="map-view-content">
      <!-- <el-menu
        default-active="2"
        class="el-menu-vertical"
        @select="chooseMode"
      >
        <el-menu-item :index="MapMode.SHOW_POINTS">
          <el-tooltip
            :content="MapModeTooltip[MapMode.SHOW_POINTS]"
            placement="right"
            >
            <img src="@/assets/连接流.svg" alt="" />
          </el-tooltip>
        </el-menu-item>
        <el-menu-item :index="MapMode.SELECT">
          <el-tooltip
            :content="MapModeTooltip[MapMode.SELECT]"
            placement="right"
            >
            <el-icon><setting /></el-icon>
          </el-tooltip>
        </el-menu-item>
      </el-menu> -->
      <div class="side-bar">
        <div class="side-bar-button" @click="toggleClusterLayer($event)">
          <el-icon><setting /></el-icon>
        </div>
      </div>
      <map-comp class="map-comp"></map-comp>
    </div>
  </div>
</template>

<script>
/* eslint-disable */
import { defineComponent } from "vue";
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
    const mapStore = useStore();

    const chooseMode = (index) => {
      console.log("chooseMode", index);
    };

    const toggleClusterLayer = () => {
      console.log('mapStore', mapStore.state, !mapStore.state.clusterLayerShow)

      console.log('toggleClusterLayer')
      mapStore.commit('setClusterLayerShow', !mapStore.state.clusterLayerShow)
    }

    return {
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
  height: 20px;
  width: 20px;
}
img:hover {
  fill: aqua;
}
.map-view {
  --menu-width: 50px;
  height: 500px;
  width: 800px;
  background-color: white;
}

.map-view-content {
  height: calc(100% - var(--header-height));
  display: flex;
}

.side-bar {
  height: calc(100% - var(--header-height));
  width: var(--menu-width);
  position: absolute;
}

.side-bar-button {
  display: flex;
  height: 40px;
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
