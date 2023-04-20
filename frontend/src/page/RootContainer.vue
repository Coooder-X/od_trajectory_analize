<template>
  <div class="root-container">
    <div class="main">
      <global-view class="global-view"></global-view>
      <map-view class="map-view"></map-view>
      <force-view></force-view>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import MapView from '@/components/views/MapView.vue';
import GlobalView from '@/components/views/GlobalView.vue'
import ForceView from '@/components/views/ForceView.vue';

export default defineComponent({
  name: 'RootContainer',
  components: {
    MapView,
    GlobalView,
    ForceView,
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
  height: 1000px;
  width: 100%;
  background-color: rgb(230, 230, 230);
  border-width: 2px;
  border-color: var(--view-border-color);
  border-style: solid;
}
</style>
