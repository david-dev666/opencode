import { Schema } from "effect"
import { ModelID, ProviderID } from "./ids.js"
import {
  mergeModelOptions,
  type ChatCompatibility,
  type ModelCapabilities,
  type ModelOptions,
  type ProviderOptions,
} from "./options.js"
import type { Auth } from "../route/auth.js"
import type { AnyRoute, CompactionOperations } from "../route/client.js"
import type { DecisionCompatibility, DecisionModelRoute } from "../decision.js"
import type { ImageRoute } from "../image.js"
import type { SpeechRoute } from "../speech.js"
import type { TranscriptionRoute } from "../transcription.js"
import type { VideoRoute } from "../video.js"
import type { ExtractionRoute } from "../experimental/extraction.js"

/** The kind of model, named after models.dev model types. Each `ai.*` entry point takes one. */
export type ModelType = "chat" | "image" | "speech" | "transcription" | "video" | "extraction" | "decision"

/** Token limits, from models.dev `limit`. Carried for callers; requests don't read them yet. */
export interface ModelLimit {
  readonly context?: number
  readonly input?: number
  readonly output?: number
}

/** Input and output modalities, from models.dev `modalities`. Carried for callers; requests don't read them yet. */
export interface ModelModalities {
  readonly input?: ReadonlyArray<string>
  readonly output?: ReadonlyArray<string>
}

type Compact = CompactionOperations | undefined

/**
 * One model of any type: who serves it, how it is reached, what it accepts, and the options every request starts
 * from. `type` selects the `ai.*` entry point that takes it.
 */
export class Model<
  Type extends ModelType = ModelType,
  Options extends ProviderOptions = ProviderOptions,
  C extends Compact = Compact,
> {
  // The API is erased over `Options`; constructors trust that it accepts every request this model's `Options` admit.
  declare protected readonly _ProviderOptions: Options
  readonly id: ModelID
  readonly provider: ProviderID
  readonly type: Type
  /** How requests are built and sent. Text APIs may carry a default `baseURL` and `auth`, which the fields below override. */
  readonly api: Model.Api<Type, C>
  /** Text models only: where requests go. */
  readonly baseURL?: string
  /** Text models only: how requests authenticate. */
  readonly auth?: Auth.Definition
  readonly options?: ModelOptions
  readonly capabilities?: ModelCapabilities
  readonly limit?: ModelLimit
  readonly modalities?: ModelModalities
  readonly compatibility?: Model.Compatibility<Type>
  /** @internal The API deployed at `baseURL` with `auth`; what the clients run. */
  readonly route: Model.Api<Type, C>

  private constructor(input: Model.Fields<Type, C>, route: Model.Api<Type, C>) {
    this.id = input.id
    this.provider = input.provider
    this.type = input.type
    this.api = input.api
    this.baseURL = input.baseURL
    this.auth = input.auth
    this.options = input.options
    this.capabilities = input.capabilities
    this.limit = input.limit
    this.modalities = input.modalities
    this.compatibility = input.compatibility
    this.route = route
  }

  static make<Type extends ModelType, Options extends ProviderOptions = ProviderOptions, C extends Compact = Compact>(
    input: Model.Input<Type, C>,
  ): Model<Type, Options, C> {
    const fields = {
      id: ModelID.make(input.id),
      provider: ProviderID.make(input.provider),
      type: input.type,
      api: input.api,
      options: input.options,
      capabilities: input.capabilities,
      limit: input.limit,
      modalities: input.modalities,
      compatibility: input.compatibility,
    }
    if (input.type !== "chat") return new Model<Type, Options, C>(fields, input.api)
    const api = input.api as AnyRoute<C>
    const baseURL = input.baseURL ?? api.endpoint.baseURL
    const auth = input.auth ?? api.auth
    const route =
      baseURL === api.endpoint.baseURL && auth === api.auth ? api : api.with({ endpoint: { baseURL }, auth })
    return new Model<Type, Options, C>(
      // The API's options are a preset under the model's own.
      { ...fields, baseURL, auth, options: mergeModelOptions(api.options, input.options) },
      route as Model.Api<Type, C>,
    )
  }

  /** A copy with `patch` applied; fields `patch` leaves out are kept. A new `api` brings its own `baseURL` and `auth`. */
  static with<Options extends ProviderOptions, Next extends Compact>(
    model: Model<"chat", Options>,
    patch: Partial<Model.Input<"chat", Next>> & { readonly api: AnyRoute<Next> },
  ): Model<"chat", Options, Next>
  static with<Type extends ModelType, Options extends ProviderOptions, C extends Compact>(
    model: Model<Type, Options, C>,
    patch: Partial<Omit<Model.Input<Type, C>, "api">> & { readonly api?: undefined },
  ): Model<Type, Options, C>
  static with<Type extends ModelType, Options extends ProviderOptions>(
    model: Model<Type, Options>,
    patch: Partial<Model.Input<Type>>,
  ): Model<Type, Options>
  static with<Type extends ModelType, Options extends ProviderOptions>(
    model: Model<Type, Options>,
    patch: Partial<Model.Input<Type>>,
  ) {
    if (Object.keys(patch).length === 0) return model
    const base = patch.api === undefined ? model : { ...model, baseURL: undefined, auth: undefined }
    return Model.make<Type, Options>({ ...base, ...patch, api: patch.api ?? model.api })
  }

  /** Whether `value` is a model of the given type. */
  static is<Type extends ModelType>(type: Type) {
    return (value: unknown): value is Model<Type> => value instanceof Model && value.type === type
  }

  /** A Schema that accepts models of the given type, for request schemas. */
  static schema<Type extends ModelType>(type: Type) {
    return Schema.declare(Model.is(type), { expected: `Model<"${type}">` })
  }
}

export namespace Model {
  interface Apis<C extends Compact> {
    readonly chat: AnyRoute<C>
    readonly image: ImageRoute
    readonly speech: SpeechRoute
    readonly transcription: TranscriptionRoute
    readonly video: VideoRoute
    readonly extraction: ExtractionRoute
    readonly decision: DecisionModelRoute
  }

  interface Compatibilities {
    readonly chat: ChatCompatibility
    readonly decision: DecisionCompatibility
  }

  export type Api<Type extends ModelType, C extends Compact = Compact> = Apis<C>[Type]

  /** Wire quirks of the model's API. Only text and decision APIs have any. */
  export type Compatibility<Type extends ModelType> = Type extends keyof Compatibilities ? Compatibilities[Type] : never

  export interface Fields<Type extends ModelType, C extends Compact = Compact> {
    readonly id: ModelID
    readonly provider: ProviderID
    readonly type: Type
    readonly api: Api<Type, C>
    readonly baseURL?: string
    readonly auth?: Auth.Definition
    readonly options?: ModelOptions
    readonly capabilities?: ModelCapabilities
    readonly limit?: ModelLimit
    readonly modalities?: ModelModalities
    readonly compatibility?: Compatibility<Type>
  }

  export interface Input<Type extends ModelType, C extends Compact = Compact>
    extends Omit<Fields<Type, C>, "id" | "provider"> {
    readonly id: string | ModelID
    readonly provider: string | ProviderID
  }
}

/** The `providerOptions` type a model's requests accept. */
export type ModelProviderOptions<SelectedModel> =
  SelectedModel extends Model<ModelType, infer Options> ? Options : never
