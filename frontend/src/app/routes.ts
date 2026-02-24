import { createBrowserRouter } from "react-router";
import Home from "./pages/Home";
import ChunkingPage from "./pages/ChunkingPage";
import IndexPage from "./pages/IndexPage";
import EvalPage from "./pages/EvalPage";
import ComponentEvalPage from "./pages/ComponentEvalPage";
import Layout from "./components/Layout";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Home },
      { path: "chunking", Component: ChunkingPage },
      { path: "index", Component: IndexPage },
      { path: "eval", Component: EvalPage },
      { path: "component-eval", Component: ComponentEvalPage },
    ],
  },
]);