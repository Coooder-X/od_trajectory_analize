<template>
  <div class="root-container">
    <div class="main">
      <global-view class="global-view"></global-view>
      <map-view class="map-view"></map-view>
      <force-view></force-view>
      <trajectory-view></trajectory-view>
      <feature-view></feature-view>
      <result-view></result-view>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import MapView from '@/components/views/MapView.vue';
import GlobalView from '@/components/views/GlobalView.vue'
import ForceView from '@/components/views/ForceView.vue';
// import ForceView4vis from '@/components/views/ForceView4vis.vue';
import TrajectoryView from '@/components/views/TrajectoryView.vue';
import FeatureView from '@/components/views/FeatureView.vue';
import ResultView from '@/components/views/ResultView.vue';

export default defineComponent({
  name: 'RootContainer',
  components: {
    MapView,
    GlobalView,
    ForceView,
    // ForceView4vis,
    TrajectoryView,
    FeatureView,
    ResultView,
  },
  setup() {
    
    return {
      
    };
  },
});
</script>

<style scoped>
.root-container {
  position: absolute;
  height: 900px;
  width: 1850px;
}

.global-view {
  position: relative;
  margin-bottom: 10px;
}

.map-view {
  position: relative;
}

.force-view {
  position: relative;
  margin-bottom: 10px;
  margin-left: 10px;
  left: 850px;
  top: calc(-500px - 200px - 10px);
}

.main {
  height: 1100px;
  width: 100%;
  background-color: rgb(230, 230, 230);
  border-width: 2px;
  border-color: var(--view-border-color);
  border-style: solid;
}
</style>
