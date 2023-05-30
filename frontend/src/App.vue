<template>
  <RootContainer style="margin-top: 30px;"/>
</template>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import RootContainer from '@/page/RootContainer.vue';

@Options({
  components: {
    RootContainer,
  },
})
export default class App extends Vue {}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}
</style>
